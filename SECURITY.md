# Security Guidelines - Jellyfin.Ollama.Chat

## Overview

This document outlines the security measures and best practices implemented in the Jellyfin.Ollama.Chat RAG system. **Please review these guidelines before deploying to production.**

## Critical Security Configurations

### 1. Environment Variables (Production Required)

**Required Environment Variables for Production:**

```bash
# JWT Security
JWT_SECRET_KEY=<strong-32+-character-secret>
JWT_ALGORITHM=RS256

# Redis Security  
REDIS_PASSWORD=<strong-redis-password>

# API Security
API_KEY_ENABLED=true
API_KEY=<strong-api-key>

# Jellyfin Integration
JELLYFIN_API_KEY=<your-jellyfin-api-key>
JELLYFIN_URL=https://your-jellyfin-server:8096
```

### 2. Redis Security Hardening

**Production Redis Configuration:**

```redis
# Enable authentication
requirepass ${REDIS_PASSWORD}

# Bind to specific interface
bind 127.0.0.1

# Enable protected mode
protected-mode yes

# Disable dangerous commands (already configured)
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command EVAL ""
rename-command DEBUG ""
```

### 3. Plugin Security

**Plugin System Security Features:**

- ✅ Module import validation prevents arbitrary code execution
- ✅ Only allows imports from `src.plugins.*` namespace
- ✅ Blocks dangerous module names and path traversal
- ✅ Plugin data namespacing prevents data collision

**Security Validation Rules:**
- No `../` path traversal
- No system module imports (`os`, `sys`, `subprocess`)
- No code execution functions (`eval`, `exec`)
- Alphanumeric module names only

### 4. Input Validation

**API Input Security:**

- ✅ Query content filtering for dangerous patterns
- ✅ Context size limits (10KB) to prevent memory exhaustion
- ✅ Field length restrictions
- ✅ Content validation for XSS/injection prevention

**Blocked Content Patterns:**
- `<script` tags
- `javascript:` URLs
- `eval()` and `exec()` calls
- `__import__` statements
- System module access

## Security Checklist for Production

### Before Deployment:

- [ ] Set strong `JWT_SECRET_KEY` (32+ characters)
- [ ] Enable Redis password authentication
- [ ] Configure Redis to bind to specific interface
- [ ] Set strong Redis password
- [ ] Enable API key authentication
- [ ] Use HTTPS for all external communications
- [ ] Change default Grafana credentials
- [ ] Review and disable debug logging
- [ ] Implement rate limiting (recommended)
- [ ] Set up security monitoring

### Network Security:

- [ ] Use TLS/SSL for Redis connections
- [ ] Configure proper firewall rules
- [ ] Use secure networks for Docker deployment
- [ ] Implement network segmentation
- [ ] Validate all external URL access

### Monitoring and Logging:

- [ ] Enable security event logging
- [ ] Set up alerts for failed authentication
- [ ] Monitor for suspicious query patterns
- [ ] Track plugin execution metrics
- [ ] Implement log rotation and retention

## Security Features Implemented

### 🛡️ Authentication & Authorization
- JWT token-based authentication (RS256)
- API key validation
- Configurable token expiration

### 🔒 Input Validation & Sanitization
- Query content filtering
- Context size limits
- Field validation with regex patterns
- XSS/injection prevention

### 🔐 Plugin System Security
- Module import validation
- Path traversal prevention
- Code execution blocking
- Namespace isolation

### 📊 Data Protection
- Plugin data namespacing
- Secure configuration management
- Environment variable usage
- No hardcoded secrets

### 🌐 Network Security
- Redis authentication
- Protected mode enabled
- Interface binding restrictions
- Dangerous command blocking

## Vulnerability Reporting

If you discover a security vulnerability, please:

1. **DO NOT** create a public issue
2. Email security concerns to the development team
3. Include detailed reproduction steps
4. Allow time for assessment and patching

## Security Audit History

- **January 2025**: Comprehensive security audit completed
  - Plugin system hardened against code injection
  - Input validation enhanced
  - Redis configuration secured
  - JWT security improved

## Additional Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Redis Security Guidelines](https://redis.io/docs/management/security/)
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

---

**⚠️ IMPORTANT**: This system handles sensitive data and integrates with external services. Always follow security best practices and conduct regular security reviews.