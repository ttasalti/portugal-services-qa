import json

from ptqa import fetch

AIMA_HTML = """
<html><body><h1>Autorização de Residência</h1>
<ul uk-accordion>
  <li class="uk-accordion-item">
    <a class="uk-accordion-title">Documentos Necessários</a>
    <div class="uk-accordion-content"><ul>
      <li>Passaporte válido;</li>
      <li><a href="#">Declaração da morada</a>, referindo em que termos reside
        <ul><li>Contrato de arrendamento</li></ul></li>
    </ul></div>
  </li>
  <li class="uk-accordion-item">
    <a class="uk-accordion-title">Taxas</a>
    <div class="uk-accordion-content"><p>Tabela de taxas</p></div>
  </li>
</ul></body></html>
"""


def test_aima_accordion_is_extracted_with_headings():
    title, text = fetch.extract(AIMA_HTML, "https://aima.gov.pt/pt/viver/x")
    assert title == "Autorização de Residência"
    assert text.splitlines()[0] == "## Documentos Necessários"
    assert "- Passaporte válido;" in text
    assert "- Declaração da morada, referindo em que termos reside" in text
    assert "- Contrato de arrendamento" in text
    assert "## Taxas\nTabela de taxas" in text


def test_govpt_service_reads_next_data():
    content = {
        "serviceTitle": {"description": "Pedir o NIF."},
        "who": {
            "position": 1,
            "title": "++**Quem**++ pode pedir",
            "descriptions": [{"description": "Qualquer pessoa."}],
        },
        "how": {
            "position": 2,
            "serviceChannels": [
                {
                    "title": "Online",
                    "verticalTabs": [
                        {
                            "title": "Quanto custa?",
                            "descriptions": [{"description": "Gratuito no [Portal](http://x)."}],
                        }
                    ],
                }
            ],
        },
    }
    data = {"props": {"pageProps": {"serverSideContent": content}}}
    html = f'<h1>NIF</h1><script id="__NEXT_DATA__">{json.dumps(data)}</script>'
    title, text = fetch.extract(html, "https://www.gov.pt/servicos/nif")
    assert title == "NIF"
    assert "## Quem pode pedir\nQualquer pessoa." in text
    assert "## Online: Quanto custa?\nGratuito no Portal." in text


def test_read_manual_page(tmp_path):
    path = tmp_path / "niss.txt"
    path.write_text(
        "url: https://example.pt/niss\ntitle: NISS\nagencies: social\n\n## Como pedir\nOnline.\n",
        encoding="utf-8",
    )
    page = fetch.read_manual(path)
    assert page == {
        "url": "https://example.pt/niss",
        "title": "NISS",
        "agencies": ["social"],
        "text": "## Como pedir\nOnline.",
    }
