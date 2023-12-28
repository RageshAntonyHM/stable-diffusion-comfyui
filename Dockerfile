# Import necessary base images
#FROM runpod/stable-diffusion:models-1.0.0 as sd-models
#FROM runpod/stable-diffusion-models:2.1 as hf-cache
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as runtime
#FROM scripts
#FROM proxy

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set working directory and environment variables
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Set up system
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv libpython3.10-dev ffmpeg -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs -y 
RUN git-lfs install

RUN echo "base Packages installed "

# Set up Python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install necessary Python packages
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --upgrade setuptools && \
    pip install --upgrade wheel
#RUN pip install --upgrade --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

#RUN git clone https://github.com/openai/triton.git && cd triton/python && pip install cmake && pip install -e

RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions triton  xformers==0.0.21 gdown



# install other packs
#RUN pip install --upgrade --no-cache-dir lpips basicsr insightface==0.7.3 onnx>=1.14.0 onnxruntime-gpu opencv-python>=4.7.0.72 numpy matplotlib segment-anything scikit-image piexif transformers

RUN echo "python Packages installed "


# Set up Jupyter Notebook
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension

RUN echo "jupyter Packages installed "

RUN mkdir workspace

WORKDIR /workspace

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install -r requirements.txt

RUN echo "ComfyUI Packages installed "

# Create necessary directories and copy necessary files
RUN set -e && mkdir -p /root/.cache/huggingface && mkdir /comfy-models
#COPY --from=hf-cache /root/.cache/huggingface /root/.cache/huggingface
#COPY --from=sd-models /SDv1-5.ckpt /comfy-models/v1-5-pruned-emaonly.ckpt
#COPY --from=sd-models /SDv2-768.ckpt /comfy-models/SDv2-768.ckpt
RUN wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -O ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors 
RUN wget "https://civitai.com/api/download/models/245598?type=Model&format=SafeTensor&size=pruned&fp=fp16" -O ComfyUI/models/checkpoints/Realistic_Vision_6_B1.safetensors
RUN wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -O ComfyUI/models/checkpoints/sd_xl_refiner_1.0.safetensors

RUN echo "necessary directories installed "

RUN cd ComfyUI/custom_nodes && pwd

RUN cd ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git 

RUN cd ComfyUI/custom_nodes && git clone https://github.com/Gourieff/comfyui-reactor-node.git && cd comfyui-reactor-node && python install.py

RUN cd ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && cd ComfyUI-Impact-Pack && python install.py

RUN cd ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

RUN cd ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_essentials.git

RUN cd ComfyUI/custom_nodes && git clone https://github.com/mav-rik/facerestore_cf.git && cd facerestore_cf && chmod 777 install.sh && sh install.sh
RUN cd ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && cd comfyui_controlnet_aux && pip install -r requirements.txt
RUN pip uninstall opencv-python opencv-python-headless opencv-contrib-python --yes
RUN pip install opencv-python==4.7.0.72 gitpython segment_anything 

RUN echo "GIT packs installed"

#download models
#RUN cd ComfyUI/models/controlnet && git clone https://huggingface.co/lllyasviel/ControlNet-v1-1
#RUN cd ComfyUI/models/controlnet && wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/d1b278d0d1103a3a7c4f7c2c327d236b082a75b1/diffusers_xl_canny_full.safetensors?download=true -O diffusers_xl_canny_full.safetensors
#RUN cd ComfyUI/models/controlnet && wget https://huggingface.co/lllyasviel/sd_control_collection/resolve/d1b278d0d1103a3a7c4f7c2c327d236b082a75b1/diffusers_xl_depth_full.safetensors?download=true -O diffusers_xl_depth_full.safetensors
RUN cd ComfyUI/models && mkdir ipadapter && cd ipadapter && wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors  
RUN cd ComfyUI/models/ipadapter && wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors
RUN cd ComfyUI/models/clip_vision && wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -O  CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
RUN cd ComfyUI/models/ipadapter && wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin
RUN cd ComfyUI/models/ipadapter && wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors
RUN cd ComfyUI/models/loras && wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors
RUN cd ComfyUI/models/facerestore_models && wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
#RUN cd ComfyUI/models/checkpoints  && wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -O sd_xl_base_1.0.safetensors
#RUN cd ComfyUI/models/checkpoints  && wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -O sd_xl_refiner_1.0.safetensors

# NGINX Proxy
#COPY --from=proxy nginx.conf /etc/nginx/nginx.conf
#COPY --from=proxy readme.html /usr/share/nginx/html/readme.html

COPY container-template/proxy/nginx.conf /etc/nginx/nginx.conf
COPY container-template/proxy/readme.html /usr/share/nginx/html/readme.html


RUN echo "ngnix installed "

# Copy the README.md
COPY README.md /usr/share/nginx/html/README.md


RUN apt-get update

# Start Scripts
ADD pre_start.sh /pre_start.sh
#COPY --from=scripts start.sh /
COPY container-template/start.sh /start.sh
COPY post_start.sh /post_start.sh
RUN chmod +x /start.sh
RUN chmod +x /post_start.sh
RUN CUDA_VISIBLE_DEVICES=1
RUN echo "EVERYTHING DONE"


CMD [ "/start.sh" ]
