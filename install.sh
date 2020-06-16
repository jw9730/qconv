apt-get install -y build-essential wget python3 python3-pip python3-dev git libssl-dev \
	vim tmux wget autoconf automake libtool curl make g++ unzip language-pack-en \
	ffmpeg libsm6 libxrender-dev libopenblas-dev

python3 -m pip install --upgrade pip
python3 -m pip install setuptools numpy==1.16.4 opencv-python cython tensorflow-gpu==1.15.2 networkx
python3 -m pip uninstall -y setuptools
python3 -m pip install setuptools

