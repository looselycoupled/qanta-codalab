
import click
from tqdm import tqdm
from flask import Flask, jsonify, request

from qanta.tfidf import TfidfGuesser
from qanta.models.dan import DanGuesser, DanModel, DanEncoder

BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(tfidf_model, dan_model, question_text):
    tfidf_guesses = tfidf_model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    dan_guesses = dan_model.guess(question_text, BUZZ_NUM_GUESSES)

    question_len = len(question_text.split(" "))
    print(question_len)

    if question_len < 50:
        print("INFO: replying with TFIDF")
        scores = [guess[1] for guess in tfidf_guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        return tfidf_guesses[0][0], buzz

    print("INFO: replying with DAN")
    return dan_guesses, True



def batch_guess_and_buzz(tfidf_model, dan_model, questions):
    results = []
    for question_text in questions:
        results.append(guess_and_buzz(tfidf_model, dan_model, question_text))
    return results



# def batch_guess_and_buzz(model, questions):
#     question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
#     outputs = []
#     for guesses in question_guesses:
#         scores = [guess[1] for guess in guesses]
#         buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
#         outputs.append((guesses[0][0], buzz))
#     return outputs




def create_app(enable_batch=True, stem=True):
    tfidf_guesser = TfidfGuesser.load(stem=stem)
    dan_guesser = DanGuesser()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, dan_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, dan_guesser, questions)
        ])


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)



if __name__ == '__main__':
    cli()
