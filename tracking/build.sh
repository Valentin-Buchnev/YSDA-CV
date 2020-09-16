set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk
    python3 -m pip install --upgrade pip
    pip3 install -qq \
        pytest \
        scikit-image \
        scikit-learn \
        matplotlib \
        h5py \
        keras \
        tensorflow \
        pandas \
        tqdm \
        moviepy
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
    python3 -c 'from tensorflow.keras.applications import ResNet50, InceptionV3, Xception;\
                model = ResNet50(weights="imagenet", include_top=False);\
                model = InceptionV3(weights="imagenet", include_top=False);\
                model = Xception(weights="imagenet", include_top=False);'
}

"$@"
