# The egress proxy for SWE-Bench Pro-Corrected. Built by common/sealed_docker.py, which generates
# `filter` from allowlist.txt and tags the image with the policy sha256.
FROM alpine:3.20
RUN apk add --no-cache tinyproxy
COPY tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY filter /etc/tinyproxy/filter
EXPOSE 8888
CMD ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]
