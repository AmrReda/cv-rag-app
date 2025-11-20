# AWS Infra (Terraform)
## Quick Start
```
cd infra/aws
terraform init
terraform plan -var="db_username=admin" -var="db_password=ChangeMe!"
terraform apply -auto-approve -var="db_username=admin" -var="db_password=ChangeMe!"
```
