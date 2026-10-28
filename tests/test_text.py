from ptqa import text


def test_split_sections_keeps_heading_with_body():
    page = "Intro line\n\n## Documentos\n- Passaporte\n\n## Taxas\nGratuito"
    assert text.split_sections(page) == [
        ("", "Intro line"),
        ("Documentos", "- Passaporte"),
        ("Taxas", "Gratuito"),
    ]


def test_detect_lang():
    assert text.detect_lang("Pasaport ve vize gerekir, başvuru için randevu alın.") == "tr"
    assert text.detect_lang("Precisa do passaporte e da declaração de morada.") == "pt"
    assert text.detect_lang("You need the passport and a valid visa.") == "en"


def test_near_copy_flags_reused_wording():
    sections = [("Qual a validade das certidões de nascimento?", "Um ano após a emissão.")]
    assert text.near_copy("Qual é a validade das certidões de nascimento?", sections)
    assert not text.near_copy("A certidão que tirei há dois anos ainda serve?", sections)
