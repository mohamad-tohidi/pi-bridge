#!/usr/bin/env node
/**
 * Pi-Bridge Server
 * Bridges Python <-> pi-coding-agent SDK via JSONL stdin/stdout.
 *
 * Protocol:
 *   stdin:  JSON lines (init msg, then commands)
 *   stdout: JSON lines (ready, events, responses)
 */

import { createInterface } from 'readline';
import { mkdtempSync, writeFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { getModel } from '@earendil-works/pi-ai';
import { Type } from 'typebox';
import {
    AuthStorage,
    ModelRegistry,
    SessionManager,
    SettingsManager,
    DefaultResourceLoader,
    createAgentSession,
    defineTool,
    getAgentDir,
} from '@earendil-works/pi-coding-agent';

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const emit = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');

/** Map base_url hostname -> known provider name */
function resolveProvider(baseUrl) {
    try {
        const host = new URL(baseUrl).hostname;
        const builtins = {
            'api.anthropic.com': 'anthropic',
            'api.openai.com': 'openai',
            'api.deepseek.com': 'deepseek',
            'api.groq.com': 'groq',
            'api.x.ai': 'xai',
        };
        return builtins[host] ?? host;
    } catch {
        return 'anthropic';
    }
}

/** Map api_format string -> pi-ai api field */
const FORMAT_TO_API = {
    completion: 'openai-completions',
    response: 'openai-responses',
    anthropic: 'anthropic-messages',
};

/** Valid thinking levels */
const VALID_LEVELS = new Set(['off', 'minimal', 'low', 'medium', 'high', 'xhigh']);

// ---------------------------------------------------------------------------
// JSON Schema -> TypeBox conversion
// ---------------------------------------------------------------------------

function jsonSchemaToTypebox(schema) {
    const convert = (s) => {
        if (!s || typeof s !== 'object') {
            throw new Error(`Invalid schema node: ${JSON.stringify(s)}`);
        }
        for (const k of ['anyOf', 'oneOf', 'allOf', '$ref']) {
            if (k in s) throw new Error(`Unsupported JSON Schema keyword: ${k}`);
        }
        const { type, description, enum: enumValues, items, properties, required } = s;
        const opts = {};
        if (description) opts.description = description;
        if (enumValues && Array.isArray(enumValues)) {
            return Type.Union(enumValues.map(v => Type.Literal(v)), opts);
        }
        switch (type) {
            case 'string':  return Type.String(opts);
            case 'number':  return Type.Number(opts);
            case 'integer': return Type.Integer(opts);
            case 'boolean': return Type.Boolean(opts);
            case 'null':    return Type.Null(opts);
            case 'array': {
                const itemSchema = items ? convert(items) : Type.Unknown();
                return Type.Array(itemSchema, opts);
            }
            case 'object': {
                const requiredSet = new Set(required ?? []);
                const props = {};
                for (const [k, v] of Object.entries(properties ?? {})) {
                    const converted = convert(v);
                    props[k] = requiredSet.has(k) ? converted : Type.Optional(converted);
                }
                return Type.Object(props, opts);
            }
            default:
                return Type.Unknown(opts);
        }
    };
    return convert(schema);
}

// ---------------------------------------------------------------------------
// Build model object
// ---------------------------------------------------------------------------

function buildModel(providerName, modelConfig, baseUrl) {
    const apiField = FORMAT_TO_API[modelConfig.api_format];
    if (!apiField) {
        throw new Error(`Unsupported api_format: "${modelConfig.api_format}". Valid values: completion, response, anthropic`);
    }

    let normalizedBaseUrl = baseUrl;
    if (apiField.startsWith('openai-')) {
        const trimmed = baseUrl.replace(/\/+$/, '');
        normalizedBaseUrl = trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
    }

    let model = getModel(providerName, modelConfig.name);
    if (model) {
        model = { ...model, api: apiField, baseUrl: normalizedBaseUrl, provider: providerName };
    } else {
        model = {
            id: modelConfig.name,
            name: modelConfig.name,
            api: apiField,
            provider: providerName,
            baseUrl: normalizedBaseUrl,
            reasoning: ['high', 'xhigh', 'medium', 'low', 'minimal'].includes(modelConfig.thinking),
            input: ['text'],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 200000,
            maxTokens: 16384,
        };
    }
    return model;
}

// ---------------------------------------------------------------------------
// Async line reader
// ---------------------------------------------------------------------------

async function* readLines() {
    const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
    for await (const line of rl) {
        const trimmed = line.trim();
        if (trimmed) yield trimmed;
    }
}

const lines = readLines();

// ---------------------------------------------------------------------------
// Read and validate init message
// ---------------------------------------------------------------------------

const initLine = await lines.next();
if (initLine.done) {
    emit({ type: 'error', message: 'No init message received on stdin' });
    process.exit(1);
}

let initMsg;
try {
    initMsg = JSON.parse(initLine.value);
} catch (e) {
    emit({ type: 'error', message: `Invalid init JSON: ${e.message}` });
    process.exit(1);
}

if (initMsg.type !== 'init') {
    emit({ type: 'error', message: `Expected init message, got type: "${initMsg.type}"` });
    process.exit(1);
}

const {
    provider: providerConfig,
    model: modelConfig,
    cwd: initCwd,
    system_prompt = '',
    custom_tools: customToolDefs = [],
    skills: skillDefs = [],
} = initMsg;

const cwd = initCwd ?? process.cwd();
const agentDir = getAgentDir();

// Validate thinking level
const thinkingVal = modelConfig.thinking ?? 'off';
if (!VALID_LEVELS.has(thinkingVal)) {
    emit({ type: 'error', message: `Unsupported thinking value: "${thinkingVal}". Valid values: ${[...VALID_LEVELS].join(', ')}` });
    process.exit(1);
}

// Build model
const providerName = resolveProvider(providerConfig.base_url);
let piModel;
try {
    piModel = buildModel(providerName, modelConfig, providerConfig.base_url);
} catch (e) {
    emit({ type: 'error', message: e.message });
    process.exit(1);
}

// Pending custom tool resolvers: toolCallId -> { resolve, reject }
const pendingTools = new Map();

// ---------------------------------------------------------------------------
// Build custom AgentTool definitions
// ---------------------------------------------------------------------------

const customTools = [];
for (const toolDef of customToolDefs) {
    let parameters;
    try {
        parameters = jsonSchemaToTypebox(toolDef.parameters);
    } catch (e) {
        emit({ type: 'error', message: `Custom tool "${toolDef.name}" schema error: ${e.message}` });
        process.exit(1);
    }

    customTools.push(defineTool({
        name: toolDef.name,
        label: toolDef.name,
        description: toolDef.description,
        parameters,
        execute: async (toolCallId, params) => {
            emit({ type: 'tool_request', id: toolCallId, tool: toolDef.name, args: params });

            const result = await new Promise((resolve, reject) => {
                const timer = setTimeout(() => {
                    pendingTools.delete(toolCallId);
                    reject(new Error(`Tool call "${toolCallId}" timed out after 300s`));
                }, 300_000);
                pendingTools.set(toolCallId, {
                    resolve: (v) => { clearTimeout(timer); resolve(v); },
                    reject: (e) => { clearTimeout(timer); reject(e); },
                });
            });

            const contentStr = typeof result.content === 'string'
                ? result.content
                : JSON.stringify(result.content);

            if (result.is_error) {
                throw new Error(contentStr);
            }
            return {
                content: [{ type: 'text', text: contentStr }],
                details: {},
            };
        },
    }));
}

// ---------------------------------------------------------------------------
// Build skills
// The SDK Skill type requires filePath + baseDir (content is read from file).
// We write each skill's markdown content to a temp file so the SDK can load it.
// ---------------------------------------------------------------------------

let skillsTempDir = null;
const skills = [];

if (skillDefs.length > 0) {
    skillsTempDir = mkdtempSync(join(tmpdir(), 'pi-bridge-skills-'));

    for (const s of skillDefs) {
        const fileName = `${s.name.replace(/[^a-zA-Z0-9_-]/g, '_')}.md`;
        const filePath = join(skillsTempDir, fileName);
        writeFileSync(filePath, s.content ?? '', 'utf8');

        skills.push({
            name: s.name,
            description: s.description ?? '',
            filePath,
            baseDir: skillsTempDir,
            source: 'custom',
        });
    }
}

// Cleanup temp skill files on exit
process.on('exit', () => {
    if (skillsTempDir) {
        try { rmSync(skillsTempDir, { recursive: true, force: true }); } catch {}
    }
});

// ---------------------------------------------------------------------------
// Set up auth, registry, loader, session
// ---------------------------------------------------------------------------

const authStorage = AuthStorage.create();
authStorage.setRuntimeApiKey(providerName, providerConfig.api_key ?? '');

const modelRegistry = ModelRegistry.inMemory(authStorage);

const systemPrompt = system_prompt || 'You are a helpful assistant. Answer clearly and concisely.';

const loader = new DefaultResourceLoader({
    cwd,
    agentDir,
    systemPromptOverride: () => systemPrompt,
    ...(skills.length > 0 ? {
        skillsOverride: (current) => ({
            skills: [...current.skills, ...skills],
            diagnostics: current.diagnostics,
        }),
    } : {}),
});
await loader.reload();

let session;
try {
    const result = await createAgentSession({
        cwd,
        agentDir,
        model: piModel,
        thinkingLevel: thinkingVal === 'off' ? undefined : thinkingVal,
        authStorage,
        modelRegistry,
        noTools: 'builtin',
        customTools,
        resourceLoader: loader,
        sessionManager: SessionManager.inMemory(),
        settingsManager: SettingsManager.inMemory({ compaction: { enabled: false } }),
    });
    session = result.session;
} catch (e) {
    emit({ type: 'error', message: `Failed to create session: ${e.message}` });
    process.exit(1);
}

// ---------------------------------------------------------------------------
// Subscribe to session events
// ---------------------------------------------------------------------------

session.subscribe((event) => {
    switch (event.type) {
        case 'message_update': {
            const ame = event.assistantMessageEvent;
            if (ame.type === 'text_delta') {
                emit({ type: 'text_delta', delta: ame.delta });
            } else if (ame.type === 'thinking_delta') {
                emit({ type: 'thinking_delta', delta: ame.delta });
            }
            break;
        }
        case 'tool_execution_start':
            emit({
                type: 'tool_call',
                tool_call_id: event.toolCallId,
                tool_name: event.toolName,
                arguments: event.args ?? {},
            });
            break;
        case 'tool_execution_end': {
            let content = '';
            if (event.result) {
                if (typeof event.result === 'string') {
                    content = event.result;
                } else if (Array.isArray(event.result?.content)) {
                    content = event.result.content
                        .filter(c => c.type === 'text')
                        .map(c => c.text)
                        .join('');
                } else {
                    content = JSON.stringify(event.result);
                }
            }
            emit({
                type: 'tool_result',
                tool_call_id: event.toolCallId,
                tool_name: event.toolName,
                content,
                is_error: event.isError,
            });
            break;
        }
        case 'turn_end':
            emit({ type: 'turn_end' });
            break;
        case 'agent_end': {
            const msgs = session.agent.state.messages;
            const lastAssistant = [...msgs].reverse().find(m => m.role === 'assistant');
            const stopReason = lastAssistant?.stopReason ?? 'stop';
            emit({ type: 'agent_end', stop_reason: stopReason });
            break;
        }
        default:
            break;
    }
});

// ---------------------------------------------------------------------------
// Signal ready
// ---------------------------------------------------------------------------

emit({ type: 'ready' });

// ---------------------------------------------------------------------------
// Command loop
// ---------------------------------------------------------------------------

for await (const line of lines) {
    let cmd;
    try {
        cmd = JSON.parse(line);
    } catch (e) {
        emit({ type: 'error', message: `Invalid command JSON: ${e.message}` });
        continue;
    }

    switch (cmd.type) {
        case 'prompt':
            session.prompt(cmd.message).catch((e) => {
                emit({ type: 'error', message: `Prompt error: ${e.message}` });
                emit({ type: 'agent_end', stop_reason: 'error' });
            });
            break;

        case 'tool_result': {
            const pending = pendingTools.get(cmd.id);
            if (pending) {
                pendingTools.delete(cmd.id);
                pending.resolve({ content: cmd.content ?? '', is_error: cmd.is_error ?? false });
            }
            break;
        }

        case 'tool_error': {
            const pending = pendingTools.get(cmd.id);
            if (pending) {
                pendingTools.delete(cmd.id);
                pending.reject(new Error(cmd.message ?? 'Tool execution error'));
            }
            break;
        }

        case 'get_messages': {
            const messages = session.agent.state.messages.map((m) => {
                if (m.role === 'assistant') {
                    return {
                        role: 'assistant',
                        content: (m.content ?? []).map((c) => {
                            if (c.type === 'text') return { type: 'text', text: c.text };
                            if (c.type === 'thinking') return { type: 'thinking', thinking: c.thinking };
                            if (c.type === 'toolCall') return { type: 'tool_call', name: c.name, arguments: c.arguments };
                            return { type: c.type };
                        }),
                        stop_reason: m.stopReason,
                        error_message: m.errorMessage,
                    };
                }
                if (m.role === 'user') {
                    const content = Array.isArray(m.content)
                        ? m.content.filter(c => c.type === 'text').map(c => c.text).join('')
                        : String(m.content ?? '');
                    return { role: 'user', content };
                }
                if (m.role === 'toolResult') {
                    return {
                        role: 'tool_result',
                        tool_call_id: m.toolCallId,
                        tool_name: m.toolName,
                        content: (m.content ?? []).filter(c => c.type === 'text').map(c => c.text).join(''),
                        is_error: m.isError,
                    };
                }
                return { role: m.role };
            });
            emit({ type: 'response', command: 'get_messages', success: true, data: { messages } });
            break;
        }

        case 'get_state': {
            const s = session.agent.state;
            emit({
                type: 'response',
                command: 'get_state',
                success: true,
                data: {
                    message_count: s.messages?.length ?? 0,
                    is_streaming: session.isStreaming,
                    model: s.model ? { provider: s.model.provider, id: s.model.id } : null,
                    thinking_level: s.thinkingLevel,
                },
            });
            break;
        }

        case 'set_model': {
            try {
                const newProviderName = resolveProvider(cmd.provider.base_url);
                const newModel = buildModel(newProviderName, cmd.model, cmd.provider.base_url);
                await session.setModel(newModel);
            } catch (e) {
                emit({ type: 'error', message: e.message });
            }
            break;
        }

        case 'set_thinking_level': {
            const level = cmd.level;
            if (!VALID_LEVELS.has(level)) {
                emit({ type: 'error', message: `Invalid thinking level: "${level}"` });
                break;
            }
            session.setThinkingLevel(level === 'off' ? undefined : level);
            break;
        }

        case 'compact':
            session.compact().catch((e) => {
                emit({ type: 'error', message: `Compact error: ${e.message}` });
            });
            break;

        case 'abort':
            session.abort().catch(() => {});
            break;

        case 'shutdown':
            session.dispose();
            process.exit(0);
            break;

        default:
            emit({ type: 'error', message: `Unknown command type: "${cmd.type}"` });
    }
}
