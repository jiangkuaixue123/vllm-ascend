# The base contains the original d4ac431 A5 FP8/FP4 adaptation and compiled SFA.
FROM jojo/vllm-ascend:d4ac431-a5-fp8-fp4
COPY source.bundle /tmp/fusion-source.bundle
RUN git -C /opt/d4ac431/vllm-ascend fetch /tmp/fusion-source.bundle HEAD \
    && git -C /opt/d4ac431/vllm-ascend checkout --detach FETCH_HEAD \
    && rm /tmp/fusion-source.bundle
ARG SOURCE_COMMIT
LABEL source.fusion="${SOURCE_COMMIT}"
