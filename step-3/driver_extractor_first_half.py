from __future__ import print_function

from embeddings_processors.example_extract_embeddings import launcher
from tiramisu.tensorflow.core.backend import read_embeddings


def test(model_name, dataset, embedding_types):
    layers = launcher.get_layers(model_name)
    datasets = launcher.get_datasets()
    models = launcher.get_nns()

    if model_name not in models:
        print("Model \"{}\" is not available".format(model_name))
        return -1

    if dataset not in datasets:
        print("Dataset \"{}\" is not available".format(dataset))
        return -1

    layers = [l[0] for l in layers]
    job_id = launcher.run_experiment(
        model_name=model_name,
        dataset_name=dataset,
        operations=None,
        embedding_types=embedding_types
    )

    return job_id


def main():
    nns = launcher.get_nns()
    datasets = launcher.get_datasets()

    print("Available models:", nns)
    print("  Layer breakdown:")

    for nn in nns:
        print("   {}:".format(nn))
        layers = launcher.get_layers(nn)

        for l in layers:
            print("     ", l)

    print("Available datasets:", datasets)

    #  First half of the datasets
    datasets = ['structure', 'entity', 'animal', 'aquatic_bird', 'organism', 'solid', 'hunting_dog',
                'equipment', 'consumer_goods', 'mammal', 'arthropod', 'chordate', 'living_thing', 'covering',
                'carnivore', 'craft', 'invertebrate', 'vertebrate', 'self-propelled_vehicle', 'clothing', 'canine',
                'terrier', 'insect', 'implement', 'dog', 'motor_vehicle', 'physical_entity', 'vehicle', 'vessel']

    model_name = "vgg16_imagenet"

    for dataset in datasets:
        ret = test(model_name, dataset, "FN")
        if None in ret:
            print("Job crashed!")
            return

        embeddings_shape, duration, location = ret
        # read_embedding(dataset, location)

        print("Time spent extracting", duration, " (seconds)")
        print("Shapes:", embeddings_shape)
        print("output:", location)


if __name__ == "__main__":
    main()
