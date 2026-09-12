# Codex 聊天记录归档

这是当前服务器上 Codex 本地会话的脱敏快照，方便迁移到新主机后查看或继续
恢复会话。归档时间：2026-09-12。

## 内容

- 128 个当前 `sessions/**/*.jsonl` 会话；
- 7 个 `archived_sessions/*.jsonl` 会话；
- 108 个附件文件、36 个 shell snapshot；
- `session_index.jsonl`、非敏感 `config.toml`；
- 总计 281 个源文件，原始约 577.9 MB，gzip tar 约 167.1 MiB。

压缩包被拆成 4 个 GitHub 安全大小的分片：

```text
codex_chat_snapshot_20260912.tar.gz.part-00
codex_chat_snapshot_20260912.tar.gz.part-01
codex_chat_snapshot_20260912.tar.gz.part-02
codex_chat_snapshot_20260912.tar.gz.part-03
```

`CODEX_CHAT_ARCHIVE_PARTS.sha256` 是分片和原始 tar 大小的校验记录。

## 安全边界

归档不包含 `auth.json`、任何 SQLite/WAL 状态库、socket、cache、plugin state
或 model cache。OpenAI/GitHub token、Bearer token、已知 secret assignment、
私钥块和 URL userinfo 已替换为脱敏标记。新主机必须重新认证；不要把本归档当作
凭据备份。

## 新主机恢复

在仓库根目录执行：

```bash
cd /home/jiandongliu/project/avelang/codex_archive
sha256sum -c CODEX_CHAT_ARCHIVE_PARTS.sha256
cat codex_chat_snapshot_20260912.tar.gz.part-* > /tmp/codex_chat_snapshot_20260912.tar.gz

CODEX_HOME=${CODEX_HOME:-$HOME/.codex}
mkdir -p "$CODEX_HOME"
# 先备份已有 Codex 配置/会话，再解压；不要覆盖 auth.json。
tar -xzf /tmp/codex_chat_snapshot_20260912.tar.gz \
  -C "$CODEX_HOME" --strip-components=1 \
  codex_home/sessions codex_home/archived_sessions codex_home/attachments \
  codex_home/shell_snapshots codex_home/session_index.jsonl codex_home/config.toml
```

恢复后可以用 `codex resume` 查看本地会话；如果只需要阅读记录，也可以直接
解压并查看 `sessions/` 下的 JSONL。工作目录从旧主机迁移到新主机时，先确认
项目路径和 README 中的 `/home/jiandongliu/project/avelang` 约定，再决定是否
修改会话中的路径文本。

原始 `/home/jiandongliu/codex-backup.tgz` 没有直接上传：它是旧的、未完成且未
经过本次脱敏审计的归档；本目录的新快照作为恢复入口。
