import logging
import re

import pandas as pd
import pandera as pa

from demokratis_ml.data import schemata

logger = logging.getLogger("document_title_rule_model")


@pa.check_types
def predict(documents: schemata.FullConsultationDocumentV1) -> pd.Series:
    df = documents[["political_body", "document_title", "document_content_plain", "document_type"]].copy()
    df["document_title_clean"] = df["document_title"].map(_clean_document_title)

    already_labelled = len(df[~df["document_type"].isna()]) * 100 / len(df)
    logger.info("%.2f%% of documents already have labels", already_labelled)

    for canton_code, rules in _DOCUMENT_TITLE_STARTS_WITH_RULES.items():
        for document_type, keywords in rules.items():
            for keyword in keywords:
                index = df["document_type"].isna()
                keyword = keyword.lower()  # noqa: PLW2901
                index &= df["document_title_clean"].str.startswith(keyword)
                if canton_code != "<any>":
                    index &= df["political_body"] == canton_code
                df.loc[index, "document_type"] = document_type
                percentage_labelled = len(df[index]) * 100 / len(df)
                logger.info(
                    "Labelled %.2f%% by rule: canton=%s, title^=%s => type=%s",
                    percentage_labelled,
                    canton_code,
                    keyword,
                    document_type,
                )

    # Special ad-hoc rule:
    df.loc[
        (df["document_type"].isna())
        & df["document_title_clean"].str.match(r"beilage (\d+ )?zum anhörungsbericht")
        & (df["document_content_plain"].str.slice(0, 200).str.contains("Synopse +Beilage")),
        "document_type",
    ] = "SYNOPTIC_TABLE"

    logger.info(
        "Labelled %.2f%% of documents",
        len(df[~df["document_type"].isna()]) * 100 / len(df) - already_labelled,
    )
    # (df["document_type"].value_counts(dropna=False) * 100 / len(df)).plot.barh(title="Document types [%]")
    df["document_type"] = pd.Categorical(df["document_type"], categories=documents["document_type"].cat.categories)
    assert documents["document_type"].dtype == df["document_type"].dtype
    return df["document_type"]


def _clean_document_title(title: str) -> str:
    title = title.strip().lower()
    title = re.sub(r"^\d+\) ", "", title)  # leading "N) "
    title = re.sub(r"^\d+_", "", title)  # leading "N_"
    title = re.sub(r"^\d+-\d+ ", "", title)  # leading "N-N "
    title = re.sub(r"(\.pdf)+$", "", title)
    title = re.sub(r" (\[|\()pdf, (\d+ seiten?, )?\d+ kb(\]|\))", "", title)
    title = re.sub(r"_", " ", title)
    return title


_DOCUMENT_TITLE_STARTS_WITH_RULES = {
    "<any>": {
        "RECIPIENT_LIST": [
            # de
            "adressliste",
            "adressatenliste",
            "adressatenverzeichnis",
            "vernehmlassungsadressaten",
            "vernehmlassungsadressen",
            "verzeichnis der anhörungsadressaten",
            "liste der konsultationsadressatinnen und konsultationsadressaten",
            "liste vernehmlassungsadressaten",
            "verzeichnis der adressatinnen und adressaten",
            "liste der vernehmlassungsadressaten",
            "liste der vernehmlassungsadressatinnen und -adressaten",
            "liste der vernehmlassungsadressatinnen und -?adressaten",
            "adressaten vernehmlassung",
            "verzeichnis der vernehmlassungsadressaten",
            "verzeichnis der vernehmlassungsadressatinnen und -adressaten",
            "liste der adressaten",
            "beilage zum anhörungsbericht (adressatenverzeichnis)",
            "beilage 3: liste der vernehmlassungsadressaten",
            "liste der vernehmlassungsadressen",
            "vernehmlassungsliste",
            # fr
            "liste des destinataires",
        ],
        "SURVEY": [  # Or RESPONSE_FORM?
            # de
            "fragebogen",
            "beilage 2 zum anhörungsbericht (fragebogen)",
            "online-fragebogen",
            "elektronischer fragebogen",
        ],
        "LETTER": [
            # de
            "brief",
            "schreiben",
            "begleitschreiben",
            "begleitbrief",
            "einladungsschreiben",
            "vernehmlassungsbrief",
            "einladung zur vernehmlassung",
            "einladung zur externen vernehmlassung",
            "einladung zur stellungnahme",
            "einladungsbrief",
            "vernehmlassungsschreiben",
            "einladung vernehmlassung",
            "einladung",
            # fr
            "lettre de consultation",
            "courrier d'accompagnement",
            "lettre",
            "courrier",
        ],
        "REPORT": [
            # de
            "anhörungsbericht",
            "vernehmlassungsbericht",
            "erläuternder bericht",
            "bericht und antrag des regierungsrates",
            "bericht und antrag des regierungsrats",
            "bericht zur vernehmlassung",  # report in NW
            "erläuterungen",
            "vernehmlassungsbotschaft",
            "bericht",
            "vortrag",
            "ratschlagsentwurf",
            "regulierungsfolgenabschätzung",
            "erläuterungen zum vernehmlassungsentwurf",
            "erläuterungsbericht",
            "planungsbericht",
            "erläuternder bericht",
            "erlauternder bericht",
            "richtplananpassung",
            # fr
            "rapport explicatif",
            # it
            "rapporto esplicativo",
        ],
        "FINAL_REPORT": [
            # de
            "auswertungsbericht",
            # Dankesbrief is a special letter that is sent with the final report in UR and highlights some key findings.
            # Could also be classified as VARIOUS_TEXT.
            "dankesbrief",
            "ergebnis der vernehmlassung",
            "auswertung externe vernehmlassung",
            "antworten der vernehmlassung",
        ],
        "SYNOPTIC_TABLE": [
            # de
            "synopse",
            "synoptische darstellung",
            "beilage zum anhörungsbericht (synopse)",
            "beilage 1 zum anhörungsbericht (synopse)",
        ],
        "RESPONSE_FORM": [
            # de
            "antwortformular",
            "beilage zum anhörungsbericht (fragebogen)",
            "mitwirkungsformular",
            "fragen zur vernehmlassung",
        ],
        "DRAFT": [
            # de
            "vorentwurf",
            "vernehmlassungsvorlage",
            "vernehmlassungsentwurf",
            "entwurf gesetzestext",
            "verordnungsentwurf",
            "landratsvorlage (entwurf)",
            "gesetzestext",
            "gesetzesentwurf",
            "gesetz (entwurf)",
            "verordnung (entwurf)",
            "bildungsgesetz (entwurf)",
            "erlassentwurf",
            "erlass",
            "entwurf",
            "änderungserlass",
            "sbe",
            "antrag landrat",
            "dekret (entwurf)",
            "änderungsentwurf",
            # fr
            "avant-projet de loi",
            "projet de loi",
            "avant-projet",
        ],
        "VARIOUS_TEXT": [
            # de
            "medienmitteilung",
            "medieninformation",
            "information",
            "rrb",
            "zeitplan",
            "einladung informationsveranstaltung",
            "vortrag: vortrag-18.08.2021",
            "präsentation",
            "grundlagenplan",
            "einladung zur informationsveranstaltung",
            "motion",
            "publikationstext",
            "anhang",
        ],
    },
    "bs": {
        "VARIOUS_TEXT": [
            # de
            "fragenkatalog",
        ],
    },
    "gl": {
        "RESPONSE_FORM": [
            # de
            "fragenkatalog",
        ],
    },
}
