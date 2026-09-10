FROM alpine:3.20

ARG PIP_INDEX_URL
ARG COMPILE_CUSTOM_KERNELS

LABEL device="a2" os="ubuntu"

CMD ["echo", "hello from a2/ubuntu"]
