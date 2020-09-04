from .model_list import model_dict

def create_model(name,**kwargs):
    """create model by name"""
    if name is None:
        return None
    model = model_dict[name](**kwargs)
    return model

def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        return segments[0]
