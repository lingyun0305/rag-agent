/**
 * 认证工具脚本
 * 提供：登录状态检查、带认证头的请求、退出登录
 */

(function() {
    const AUTH_TOKEN_KEY = 'access_token';
    const USERNAME_KEY = 'username';
    const UID_KEY = 'uid';
    const USERINFO_KEY = 'user_info';
    const LOGIN_PAGE = '/front/login.html';

    // ===== 公开 API =====

    window.Auth = {
        /**
         * 获取当前登录 token
         */
        getToken() {
            return localStorage.getItem(AUTH_TOKEN_KEY);
        },

        /**
         * 获取当前用户名
         */
        getUsername() {
            return localStorage.getItem(USERNAME_KEY) || '';
        },

        /**
         * 获取当前用户ID
         */
        getUid() {
            return localStorage.getItem(UID_KEY) || '';
        },

        /**
         * 获取完整用户信息
         */
        getUserInfo() {
            const raw = localStorage.getItem(USERINFO_KEY);
            if (!raw) return null;
            try {
                return JSON.parse(raw);
            } catch {
                return null;
            }
        },

        /**
         * 保存用户信息
         */
        setUserInfo(info) {
            localStorage.setItem(USERINFO_KEY, JSON.stringify(info));
            if (info.username) localStorage.setItem(USERNAME_KEY, info.username);
            if (info.id) localStorage.setItem(UID_KEY, info.id);
        },

        /**
         * 清除本地认证信息
         */
        clearAuth() {
            localStorage.removeItem(AUTH_TOKEN_KEY);
            localStorage.removeItem(USERNAME_KEY);
            localStorage.removeItem(UID_KEY);
            localStorage.removeItem(USERINFO_KEY);
        },

        /**
         * 检查是否已登录，并验证 token 有效性
         */
        async checkLogin(redirectUrl = LOGIN_PAGE) {
            const token = this.getToken();
            if (!token) {
                location.href = redirectUrl;
                return false;
            }
            try {
                const res = await fetch('/api/auth/get_user_info', {
                    headers: { 'Authorization': `Bearer ${token}` }
                });
                if (res.status === 401 || res.status === 403) {
                    this.clearAuth();
                    location.href = redirectUrl;
                    return false;
                }
                if (res.ok) {
                    const data = await res.json();
                    this.setUserInfo(data);
                    this.updateUserDisplay();
                    return true;
                }
                // 其他错误状态码（如404等），也清除并跳转
                this.clearAuth();
                location.href = redirectUrl;
                return false;
            } catch (e) {
                // 网络错误时，如果有 token 暂时放行
                console.warn('Token validation failed (network error):', e.message);
                return true;
            }
        },

        /**
         * 带认证头的 fetch 封装
         */
        async fetch(url, options = {}) {
            const token = this.getToken();
            const headers = {
                ...(options.headers || {}),
            };
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
            const res = await fetch(url, {
                ...options,
                headers,
            });
            // 401 / 403 时自动跳转到登录页
            if (res.status === 401 || res.status === 403) {
                this.clearAuth();
                location.href = LOGIN_PAGE;
                throw new Error('登录已过期，请重新登录');
            }
            return res;
        },

        /**
         * 退出登录
         */
        logout(showRedirect = true) {
            this.clearAuth();
            if (showRedirect) {
                location.href = LOGIN_PAGE;
            }
        },

        /**
         * 更新页面上的用户信息显示
         */
        updateUserDisplay() {
            const userInfo = this.getUserInfo();
            const username = this.getUsername();
            const avatar = userInfo && userInfo.avatar ? userInfo.avatar : '';
            const fallbackText = username ? username.charAt(0).toUpperCase() : 'U';
            document.querySelectorAll('.sidebar-avatar').forEach(el => {
                if (avatar) {
                    el.innerHTML = '<img src="' + avatar + '" alt="avatar" style="width:100%;height:100%;object-fit:cover;">';
                } else {
                    el.textContent = fallbackText;
                }
            });
            // 同步更新聊天消息中已经渲染的用户头像
            document.querySelectorAll('.message.user .message-avatar').forEach(el => {
                if (avatar) {
                    el.innerHTML = '<img src="' + avatar + '" alt="avatar" style="width:100%;height:100%;object-fit:cover;border-radius:10px;">';
                } else {
                    el.textContent = fallbackText;
                }
            });
        },

        /**
         * 初始化：检查登录状态 + 验证 token
         */
        async init() {
            if (location.pathname.includes('/login.html')) return;
            
            // 防止重复初始化
            if (this._initialized) return;
            this._initialized = true;

            const token = this.getToken();
            if (!token) {
                location.href = LOGIN_PAGE;
                return;
            }
            // 异步验证 token 有效性
            await this.checkLogin(LOGIN_PAGE);
        }
    };

    // 页面加载时自动初始化
    document.addEventListener('DOMContentLoaded', function() {
        window.Auth.init();
    });

    // 如果 DOMContentLoaded 已经触发过了（脚本在页面底部），立即执行
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        window.Auth.init();
    }
})();
