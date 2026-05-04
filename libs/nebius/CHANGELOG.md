# Changelog

## 0.2.0

### Changed

- **Default `base_url` updated to `https://api.tokenfactory.nebius.com/v1/`**
  (Token Factory canonical domain). The legacy
  `https://api.studio.nebius.ai/v1/` remains accessible by passing
  `base_url=` to the constructor or by setting the `NEBIUS_API_BASE`
  environment variable.

  Affects `ChatNebius`, `NebiusEmbeddings`, and `NebiusRetriever`.

## 0.1.3

- Earlier releases. See git history.
