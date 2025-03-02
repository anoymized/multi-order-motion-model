## Note: our code implementation is based on Kubric pipeline:
https://github.com/google-research/kubric


For instructions, please refer to https://kubric.readthedocs.io

You need to install the Docker and then run the container:
```bash

git clone https://github.com/google-research/kubric.git
cd kubric
docker pull kubricdockerhub/kubruntu
docker run --rm --interactive \
           --user $(id -u):$(id -g) \
           --volume "$(pwd):/kubric" \
           kubricdockerhub/kubruntu \
           /usr/bin/python3 examples/helloworld.py
ls output
```

After that, try `seconder.py` or `seconder_sep.py` to test the render function.
