from src.speaker import auto_alias_map


def test_title_variants_merge_only_when_short_name_speaks():
    names = ["MAYA", "LADY MAYA", "NOVA", "LADY NOVA", "LORD HALE", "LADY HALE", "LORD"]
    assert auto_alias_map(names) == {"LADY MAYA": "MAYA", "LADY NOVA": "NOVA"}


def test_disguise_names_are_not_merged():
    assert auto_alias_map(["ORION", "MYSTERIOUS HOODED MAN", "THE HOODED STRANGER"]) == {}
