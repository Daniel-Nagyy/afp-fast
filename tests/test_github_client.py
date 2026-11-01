"""
Mocked-response tests for github_client.py.

These check the parsing logic against the documented GitHub REST API
response shape WITHOUT making a live network call -- see the module
docstring in github_client.py and the README for why: a live call needs
a real personal access token that only the user can provide, and this
build sandbox cannot attach one for a disproportionate, test-only reason.
"""

from unittest.mock import MagicMock

from afp.github_client import GitHubClient


def _mock_response(status_code=200, json_data=None, headers=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.headers = headers or {}
    resp.text = text
    return resp


def test_get_repo_parses_fields():
    client = GitHubClient(token="fake-token")
    client.session.get = MagicMock(return_value=_mock_response(
        200,
        {
            "full_name": "octocat/Hello-World",
            "stargazers_count": 42,
            "fork": False,
            "archived": False,
            "pushed_at": "2026-09-01T00:00:00Z",
            "created_at": "2011-01-26T19:01:12Z",
            "language": "Python",
            "default_branch": "main",
        },
    ))
    repo = client.get_repo("octocat", "Hello-World")
    assert repo["full_name"] == "octocat/Hello-World"
    assert repo["stargazers_count"] == 42


def test_get_repo_raises_on_error_status():
    client = GitHubClient(token="fake-token")
    client.session.get = MagicMock(return_value=_mock_response(404, {}, text="Not Found"))
    try:
        client.get_repo("nope", "nope")
        assert False, "expected GitHubAPIError"
    except Exception as e:
        assert "404" in str(e)


def test_get_branch_protection_200_true_404_false_403_unknown():
    client = GitHubClient(token="fake-token")

    client.session.get = MagicMock(return_value=_mock_response(200, {"required_status_checks": {}}))
    assert client.get_branch_protection("o", "r", "main") is True

    client.session.get = MagicMock(return_value=_mock_response(404, {}))
    assert client.get_branch_protection("o", "r", "main") is False

    client.session.get = MagicMock(return_value=_mock_response(403, {}, text="Forbidden"))
    assert client.get_branch_protection("o", "r", "main") is None


def test_get_all_pages_stops_on_short_page():
    client = GitHubClient(token="fake-token")
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": i} for i in range(100, 150)]  # short page -> last one

    responses = [_mock_response(200, page1), _mock_response(200, page2)]
    client.session.get = MagicMock(side_effect=responses)

    items = client._get_all_pages("/repos/o/r/contributors")
    assert len(items) == 150
