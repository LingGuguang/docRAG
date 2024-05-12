import argparse

def main_argparser():
    parser = argparse.ArgumentParser(description="for docRAG")
    
    parser.add_argument('--hypocritical_answer', default=False, type=bool, help='generate a hypocritical answer to recall answer.')
    parser.add_argument('--additional_query', default=0, type=int, help='generate k additional query to recall answer.')

    return parser.parse_args()


