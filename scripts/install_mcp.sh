#!/bin/sh
# 显式安装 @z_ai/mcp-server 固定版本（GLM Coding Plan MCP 适配用）。
# 运行时绝不自动下载；本脚本是唯一安装入口。版本变更需同步更新
# backend/app/model_providers/mcp_client.py 的 PINNED_VERSION 与冻结 schema 哈希。
set -eu
cd "$(dirname "$0")/../backend"
PKG=@z_ai/mcp-server
VER=0.1.5
npm install "$PKG@$VER" --no-save --no-audit --no-fund
echo "已安装 $PKG@$VER → backend/node_modules（probe 将校验 schema 哈希）"
