def pytest_addoption(parser):
    parser.addoption("--host", action="store", default="10.99.2.70", help="Robot hostname or IP address")
