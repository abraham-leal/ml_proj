variable "project_id" {
  type        = string
  description = "Project ID"
}

variable "region" {
  type        = string
  description = "Google region"
}

variable "zone" {
  type        = string
  description = "Google zone"
}

variable "namespace" {
  type        = string
  description = "Namespace prefix used for resources"
}

variable "domain_name" {
  type        = string
  description = "Domain name for accessing the Weights & Biases UI."
}

variable "subdomain" {
  type        = string
  description = "Subdomain for access the Weights & Biases UI."
}

variable "license" {
  type        = string
  description = "W&B License"
}
