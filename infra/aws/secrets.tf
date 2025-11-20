# --- OpenAI key in Secrets Manager ---
resource "aws_secretsmanager_secret" "openai" {
  name        = "${local.name}-OPENAI_API_KEY"
  description = "OpenAI API key for ${local.name}"
}

# Option A: set an initial placeholder (update in console/CLI later)
resource "aws_secretsmanager_secret_version" "openai_v1" {
  secret_id     = aws_secretsmanager_secret.openai.id
  secret_string = "replace-me-in-console"
}

# --- Allow ECS task execution role to read the secret ---
data "aws_iam_policy_document" "secrets_read" {
  statement {
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn]
  }
}

resource "aws_iam_policy" "secrets_read" {
  name   = "${local.name}-secrets-read"
  policy = data.aws_iam_policy_document.secrets_read.json
}

resource "aws_iam_role_policy_attachment" "exec_can_read_secret" {
  role       = module.ecs_task_execution_role.iam_role_name
  policy_arn = aws_iam_policy.secrets_read.arn
}
