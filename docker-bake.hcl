variable "RELEASE" {
    default = "3.0.2"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["rageshantony/comfyui_own"]
    contexts = {
        scripts = "./container-template"
        proxy = "./container-template/proxy"
    }
}
