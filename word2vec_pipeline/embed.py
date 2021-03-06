from utils.os_utils import mkdir
import model_building as mb
from utils.db_utils import item_iterator


def embed_from_config(config):

    mkdir(config["embedding"]["output_data_directory"])

    # If there is a whitelist only keep the matching filename
    try:
        whitelist = config["score"]["input_file_whitelist"]
    except:
        whitelist = []

    #
    # Run the functions that act globally on the data

    for name in config["embedding"]["embedding_commands"]:
        obj = getattr(mb, name)

        # Load any kwargs in the config file
        kwargs = config["embedding"].copy()

        if name in kwargs:
            kwargs.update(kwargs[name])
        kwargs['target_column'] = config['target_column']

        func = obj(**kwargs)
        func.set_iterator_function(item_iterator,
                                   config["embedding"],
                                   whitelist,
                                   section="parse")
        func.compute(**kwargs)


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    embed_from_config(config)
