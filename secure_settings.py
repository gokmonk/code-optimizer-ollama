import os
import json
from cryptography.fernet import Fernet
from keyring import get_password, set_password
import re
from dotenv import load_dotenv

class SecureSettings:
    def __init__(self):
        self.service_name = "code_optimizer"
        self.key_file = ".encryption_key"
        self._load_or_create_key()
        load_dotenv()

    def _load_or_create_key(self):
        """Load or create encryption key"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(self.key)
        self.cipher_suite = Fernet(self.key)

    def _validate_api_key(self, key, service):
        """Validate API key format"""
        if not key:
            return False
        
        # Basic format validation
        if service == "openai":
            return bool(re.match(r'^sk-[A-Za-z0-9]{32,}$', key))
        elif service == "anthropic":
            return bool(re.match(r'^sk-ant-[A-Za-z0-9]{32,}$', key))
        elif service == "deepseek":
            return bool(re.match(r'^[A-Za-z0-9]{32,}$', key))
        return True

    def get_api_key(self, service):
        """Get encrypted API key from keyring"""
        encrypted_key = get_password(self.service_name, service)
        if encrypted_key:
            try:
                return self.cipher_suite.decrypt(encrypted_key.encode()).decode()
            except:
                return None
        return None

    def set_api_key(self, service, key):
        """Set encrypted API key in keyring"""
        if not self._validate_api_key(key, service):
            raise ValueError(f"Invalid {service} API key format")
        
        encrypted_key = self.cipher_suite.encrypt(key.encode())
        set_password(self.service_name, service, encrypted_key.decode())

    def get_env_api_key(self, env_var):
        """Get API key from environment variable"""
        return os.getenv(env_var)

    def validate_all_keys(self):
        """Validate all stored API keys"""
        services = ['openai', 'anthropic', 'deepseek', 'chatgpt']
        results = {}
        for service in services:
            key = self.get_api_key(service)
            results[service] = {
                'exists': key is not None,
                'valid': self._validate_api_key(key, service) if key else False
            }
        return results

    def migrate_env_to_secure(self):
        """Migrate API keys from environment variables to secure storage"""
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'chatgpt': 'CHATGPT_API_KEY'
        }
        
        for service, env_var in env_vars.items():
            key = self.get_env_api_key(env_var)
            if key and not self.get_api_key(service):
                self.set_api_key(service, key)
                # Clear the environment variable after migration
                os.environ[env_var] = '' 