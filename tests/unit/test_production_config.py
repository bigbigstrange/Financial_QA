import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.core.config import Settings


class ProductionConfigTests(unittest.TestCase):
    def settings(self, **values):
        with patch.dict(os.environ, {}, clear=True):
            return Settings(_env_file=None, **values)

    def test_local_defaults_remain_usable(self):
        settings = self.settings()
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.bootstrap_admin_password, "admin123")

    def test_non_loopback_bind_requires_production(self):
        with self.assertRaises(ValidationError):
            self.settings(host="0.0.0.0")

    def test_production_cannot_enable_debug(self):
        with self.assertRaises(ValidationError):
            self.settings(
                app_env="production",
                database_url="sqlite:////app/data_runtime/data_app.db",
                jwt_secret="a-random-production-signing-secret-123456",
                bootstrap_admin_password="a-separate-admin-password-123456",
                cors_origins="",
                debug=True,
            )

    def test_production_rejects_unapproved_jwt_algorithm(self):
        with self.assertRaises(ValidationError):
            self.settings(
                app_env="production",
                database_url="sqlite:////app/data_runtime/data_app.db",
                jwt_secret="a-random-production-signing-secret-123456",
                bootstrap_admin_password="a-separate-admin-password-123456",
                cors_origins="",
                jwt_algorithm="none",
            )

    def test_production_rejects_insecure_defaults(self):
        with self.assertRaises(ValidationError):
            self.settings(app_env="production")

    def test_validation_error_does_not_echo_credentials(self):
        token = "sensitive-signing-value-for-error-check-123456"
        with self.assertRaises(ValidationError) as caught:
            self.settings(app_env="production", jwt_secret=token)
        self.assertNotIn(token, str(caught.exception))
        self.assertNotIn("input_value", str(caught.exception))

    def test_production_requires_strong_distinct_secrets(self):
        valid = {
            "app_env": "production",
            "database_url": "sqlite:////app/data_runtime/data_app.db",
            "jwt_secret": "a-random-production-signing-secret-123456",
            "bootstrap_admin_password": "a-separate-admin-password-123456",
            "cors_origins": "",
        }
        self.assertEqual(self.settings(**valid).app_env, "production")
        for field, weak in (
            ("jwt_secret", "change-me-in-production"),
            ("bootstrap_admin_password", "admin123"),
            ("bootstrap_admin_password", valid["jwt_secret"]),
        ):
            with self.subTest(field=field, weak=weak):
                with self.assertRaises(ValidationError):
                    self.settings(**{**valid, field: weak})

    def test_production_rejects_wildcard_cors_and_unpersisted_db(self):
        valid = {
            "app_env": "production",
            "database_url": "sqlite:////app/data_runtime/data_app.db",
            "jwt_secret": "a-random-production-signing-secret-123456",
            "bootstrap_admin_password": "a-separate-admin-password-123456",
            "cors_origins": "",
        }
        for override in ({"cors_origins": "*"}, {"database_url": "sqlite:///data_app.db"}):
            with self.subTest(override=override):
                with self.assertRaises(ValidationError):
                    self.settings(**{**valid, **override})
        self.assertEqual(self.settings(**{**valid, "cors_origins": ""}).cors_origin_list, [])


if __name__ == "__main__":
    unittest.main()
