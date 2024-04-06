from korbit-tutorial-8de47d48ea83bd23.util import process_work


def do_some_work(work_to_be_done: list[str]):
    finished_work = []
    for work in work_to_be_done:
        finished_work.append(process_work(work))
    return finished_work


if __name__ == "__main__":
    work_to_be_done = ["these", "are", "some", "words", None]
    print(do_some_work(work_to_be_done))
