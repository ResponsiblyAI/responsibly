# TODO how import files from a package
import json

from pkg_resources import resource_string


def load_json_resource(resource_name):
    return json.loads(
        resource_string(__name__, resource_name + '.json').decode('utf-8')

    )


BOLUKBASI_DATA = load_json_resource('bolukbasi')

BOLUKBASI_DATA['gender']['profession_names'] = list(
    zip(*BOLUKBASI_DATA['gender']['professions']))[0]


# TODO: in the code of the article, the last definitional pair
# is not in the specific full
BOLUKBASI_DATA['gender']['specific_full_with_definitional'] = (
    set.union(
        *map(set, BOLUKBASI_DATA['gender']['definitional_pairs'])
    ) | set(BOLUKBASI_DATA['gender']['specific_full'])
)


BOLUKBASI_DATA['gender']['neutral_profession_names'] = list(
    set(BOLUKBASI_DATA['gender']['profession_names'])
    - set(BOLUKBASI_DATA['gender']['specific_full_with_definitional'])
)
