# TODO how import files from a package
import json

from pkg_resources import resource_string


def load_json_resource(resource_name):
    return json.loads(
        resource_string(__name__, resource_name + '.json')
    )


TOLGA_DATA = load_json_resource('tolga')

TOLGA_DATA['gender']['professions_names'] = list(
    zip(*TOLGA_DATA['gender']['professions']))[0]
