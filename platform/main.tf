provider "google" {
 project = var.project_id
 region  = var.region
 zone    = var.zone
}

provider "google-beta" {
 project = var.project_id
 region  = var.region
 zone    = var.zone
}

data "google_client_config" "current" {}

provider "kubernetes" {
  host                   = "https://${module.wandb.cluster_endpoint}"
  cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
  token                  = data.google_client_config.current.access_token
}

# Spin up all required services
module "wandb" {
  source  = "wandb/wandb/google"
  version = "~> 1.0"

  namespace   = var.namespace
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain
  create_redis = true
  allowed_inbound_cidrs = ["*"]
}

# You'll want to update your DNS with the provisioned IP address
output "url" {
  value = module.wandb.url
}

output "address" {
  value = module.wandb.address
}

output "bucket_name" {
  value = module.wandb.bucket_name
}
