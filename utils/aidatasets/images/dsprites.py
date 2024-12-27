import numpy as np
import datasets


class DSprites(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "orientation": datasets.Value("float"),
                "shape": datasets.ClassLabel(names=["square", "ellipse", "heart"]),
                "scale": datasets.Value("float"),
                "color": datasets.ClassLabel(names=["white"]),
                "position_x": datasets.Value("float"),
                "position_y": datasets.Value("float"),
            }
        )

        homepage = "https://github.com/deepmind/dsprites-dataset"
        license = "zlib/libpng"
        return datasets.DatasetInfo(
            description="""dSprites is a dataset of 2D shapes procedurally generated from 6 ground truth independent latent factors. These factors are color, shape, scale, rotation, x and y positions of a sprite.
All possible combinations of these latents are present exactly once, generating N = 737280 total images.""",
            features=features,
            supervised_keys=("image", "shape"),
            homepage=homepage,
            license=license,
            citation="""@misc{dsprites17,
author = {Loic Matthey and Irina Higgins and Demis Hassabis and Alexander Lerchner},
title = {dSprites: Disentanglement testing Sprites dataset},
howpublished= {https://github.com/deepmind/dsprites-dataset/},
year = "2017",
}""",
        )

    def _split_generators(self, dl_manager):
        archive = dl_manager.download(
            "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive": archive},
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, archive):
        dataset_zip = np.load(archive, allow_pickle=True)
        images = dataset_zip["imgs"]
        latents_values = dataset_zip["latents_values"]
        for key in range(len(images)):
            yield key, {
                "image": images[key],
                "color": int(latents_values[key, 0]) - 1,
                "shape": int(latents_values[key, 1]) - 1,
                "scale": latents_values[key, 2],
                "orientation": latents_values[key, 3],
                "position_x": latents_values[key, 4],
                "position_y": latents_values[key, 5],
            }
