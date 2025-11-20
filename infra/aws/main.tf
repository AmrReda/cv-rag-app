locals { name = var.project }

# --- VPC ---
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = local.name
  cidr = var.vpc_cidr
  azs  = ["${var.region}a","${var.region}b"]

  public_subnets  = var.public_subnets
  private_subnets = var.private_subnets

  enable_nat_gateway = true
  single_nat_gateway = true
}

# --- S3 bucket for blobs ---
resource "aws_s3_bucket" "blobs" {
  bucket = "${local.name}-blobs"
  force_destroy = true
}
resource "aws_s3_bucket_versioning" "blobs" {
  bucket = aws_s3_bucket.blobs.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_server_side_encryption_configuration" "blobs" {
  bucket = aws_s3_bucket.blobs.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# --- RDS Postgres ---
module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.5"

  identifier = "${local.name}-pg"
  engine            = "postgres"
  engine_version    = "16.3"
  family            = "postgres16"
  instance_class    = "db.t4g.micro"
  allocated_storage = 20

  db_name  = "cv_rag"
  username = var.db_username
  password = var.db_password

  vpc_security_group_ids = [module.vpc.default_security_group_id]
  subnet_ids             = module.vpc.private_subnets
  publicly_accessible    = false
  multi_az               = false

  deletion_protection = false
  skip_final_snapshot = true
}

# --- ElastiCache Redis (for queues/caching) ---
module "redis" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 5.7"

  cluster_id = "${local.name}-redis"
  engine     = "redis"
  node_type  = "cache.t4g.micro"
  num_cache_nodes = 1
  subnet_group_name = "${local.name}-redis-subnet"
  vpc_security_group_ids = [module.vpc.default_security_group_id]
  subnet_ids = module.vpc.private_subnets
}

# --- ECS Fargate cluster ---
module "ecs" {
  source  = "terraform-aws-modules/ecs/aws"
  version = "~> 5.11"

  cluster_name = "${local.name}-cluster"
}

# --- IAM for task execution ---
module "ecs_task_execution_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-assumable-role"
  version = "~> 5.48"

  create_role = true
  role_name   = "${local.name}-exec-role"
  trusted_role_services = ["ecs-tasks.amazonaws.com"]
  custom_role_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  ]
}

# --- ECS task definition & service (API only here) ---
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu    = "512"
  memory = "1024"

  execution_role_arn = module.ecs_task_execution_role.iam_role_arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.container_image
      essential = true
      portMappings = [{ containerPort = 8000, hostPort = 8000 }]
      environment = [
        { name = "DATA_DIR", value = "/data" },
        # add OPENAI_API_KEY via secrets manager at deploy time
      ],
      secrets = [
      {
        name      = "OPENAI_API_KEY",
        valueFrom = aws_secretsmanager_secret.openai.arn
      }
    ]
    }
  ])
}

resource "aws_lb" "api" {
  name            = "${local.name}-alb"
  internal        = false
  load_balancer_type = "application"
  subnets         = module.vpc.public_subnets
  security_groups = [module.vpc.default_security_group_id]
}

resource "aws_lb_target_group" "api" {
  name     = "${local.name}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = module.vpc.vpc_id
  target_type = "ip"
  health_check {
    path = "/docs"
    port = "8000"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"
  default_action { type = "forward" target_group_arn = aws_lb_target_group.api.arn }
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api-svc"
  cluster         = module.ecs.cluster_id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.ecs_desired_count
  launch_type     = "FARGATE"
  network_configuration {
    assign_public_ip = true
    subnets         = module.vpc.public_subnets
    security_groups = [module.vpc.default_security_group_id]
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  depends_on = [aws_lb_listener.http]
}
