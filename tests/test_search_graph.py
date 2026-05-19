from scripts.search_graph import parse_args


def test_parse_args_for_search_graph(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["search_graph.py", "GraphRAG", "--top-k", "3", "--max-depth", "2"])

    args = parse_args()

    assert args.query == "GraphRAG"
    assert args.top_k == 3
    assert args.max_depth == 2
