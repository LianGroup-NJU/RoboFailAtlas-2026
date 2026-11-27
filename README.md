# RoboFailAtlas

**Structured Reward Learning from Diverse Robot Failures**

[Project website](https://failatlas-research-preview.aquaash11.chatgpt.site/)

This repository currently contains the project website, paper figures, narrated
overview video, and interactive trajectory examples. Research code will be added
in a separate update.

## Website files

The complete static website is in [`docs/`](docs/). It includes its fonts, images,
videos, and example annotations, and requires no build step or external CDN.

To preview it locally, run the following command from this repository and open
<http://localhost:8000>:

```sh
python -m http.server 8000 --directory docs
```

## GitHub Pages

A repository administrator can enable hosting at **Settings → Pages**:

1. Set **Source** to **Deploy from a branch**.
2. Select the **main** branch and **/docs** folder.
3. Save and wait for the Pages deployment to finish.

The default project URL will then be
<https://liangroup-nju.github.io/RoboFailAtlas-2026/>. Hosting availability for a
private repository depends on the organization's GitHub plan and policy.

The existing project website above remains available while GitHub Pages is being
configured. Future website updates should be committed under `docs/`.

## Release links

Confirmed paper, dataset, model, author, and citation information is configured in
[`docs/release.json`](docs/release.json). Unreleased resource buttons stay inactive
until their actual URLs are supplied.

The bundled Google Sans Flex font retains its
[SIL Open Font License](docs/assets/fonts/OFL.txt).
