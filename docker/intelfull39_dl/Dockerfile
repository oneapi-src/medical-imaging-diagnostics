FROM devkitintelfull39
ENV https_proxy=$https_proxy
ENV http_proxy=$http_proxy
ENV ftp_proxy=$ftp_proxy
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
SHELL ["/bin/bash", "-c"] 
RUN apt-get update
RUN apt-get install libgl1 -y
RUN apt-get install libglib2.0-0 -y
