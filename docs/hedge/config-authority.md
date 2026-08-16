# Hedge configuration authority

The runtime configuration has one canonical path for each responsibility:

- `freqtrade/hedge/config.py` owns runtime normalization and semantic validation.
- `freqtrade/hedge/config_schema_extension.py` is the only Hedge JSON-schema
  definition module. It extends the upstream schema at the integration hook;
  it does not perform runtime migration.
- `freqtrade/hedge/config_migration.py` performs the one-way raw-input migration
  from the retired `hedge.r56` key to `hedge.operations` before schema validation.
- `freqtrade/hedge/operations/config.py` owns the runtime operations settings
  after normalization. Other subsystem config modules are local adapters and do
  not define a second operations authority.

`user_data/` is runtime-owned. Its SQLite databases, logs, checkpoints and model
artifacts are intentionally outside the source-version naming policy; in
particular, `user_data/r5` remains preserved runtime data. Clean-mainline checks
apply only to the declared, version-controlled source set.
