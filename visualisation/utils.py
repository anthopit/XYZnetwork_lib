import distinctipy as distipy

def create_comm_colors(nb_colors):
    """
    Create a list of colors for the communities
    :param communities: list of communities
    :return: list of colors
    """
    colors = distipy.get_colors(nb_colors)
    colors = [tuple([i * 255 for i in c]) for c in colors]
    # convert rgb tuple to hex
    colors = [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in colors]

    return colors