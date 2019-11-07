from tensorflow.python import pywrap_tensorflow


def get_checkpoint_variables(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        var_to_dtype_map = reader.get_variable_to_dtype_map()
        vals = dict()
        for key, value in sorted(var_to_shape_map.items()):
            vals[key] = reader.get_tensor(key)
        return vals
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
            if ("Data loss" in str(e) and
                    any(e in file_name for e in [".index", ".meta", ".data"])):
                proposed_file = ".".join(file_name.split(".")[0:-1])
                v2_file_error_template = """
                                      It's likely that this is a V2 checkpoint and you need to provide the filename
                                      *prefix*.  Try removing the '.' and extension.  Try:
                                      inspect checkpoint --file_name = {}"""
                print(v2_file_error_template.format(proposed_file))
