variable "region" { type = string  default = "ap-southeast-2" }
variable "project" { type = string default = "cv-rag-app" }

variable "vpc_cidr" { type = string default = "10.20.0.0/16" }
variable "public_subnets"  { type = list(string) default = ["10.20.1.0/24","10.20.2.0/24"] }
variable "private_subnets" { type = list(string) default = ["10.20.11.0/24","10.20.12.0/24"] }

variable "db_username" { type = string }
variable "db_password" { type = string sensitive = true }

variable "ecs_desired_count" { type = number default = 2 }
variable "container_image" { type = string default = "ghcr.io/AmrReda/cv-rag-app:latest" }
