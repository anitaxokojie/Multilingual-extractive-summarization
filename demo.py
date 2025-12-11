from src.models import BERTSummarizer, DomainTunedSummarizer
import time

def print_section(title, content):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{content}\n")

def main():
    print(" Initializing models (this may take a moment)...")
    start_time = time.time()
    
    # Initialize the summarizers
    basic_summarizer = BERTSummarizer()
    tuned_summarizer = DomainTunedSummarizer()
    
    print(f" Models loaded in {time.time() - start_time:.2f}s\n")
    
    # Example TED Talk excerpt
    english_text = """
    Thank you so much, Chris. And it's truly a great honor to have the opportunity to come to this stage twice. I'm extremely grateful. I have been blown away by this conference, and I want to thank all of you for the many nice comments about what I had to say the other night. And I want to thank you for what you are doing. Those of you who have been here, for those of you who are watching, the people who have been here have been so actively involved in trying to make this world a better place. The brilliant presentations that we have heard, the extraordinary performances that we have enjoyed, the conversations in the hallways and at meals -- it's been an incredible experience for me.

    I'm going to focus on the climate crisis, and I have a 10-minute slide show -- 10-minute new slide show -- I update the slide show almost every day, so this is a new slide show focused on the climate crisis. But I want to begin with a couple of stories from the aftermath of the movie.

    I've been trying to tell this story for a long time. I was reminded by Tipper, who is right there, she saw the first slide show 30 years ago. And -- oh, there's a skeptic over here. There are a few people who have seen the slide show many, many times. But telling this story about the climate crisis over a long period of time -- there was a turning point in 2015 when, in December, virtually every nation in the entire world agreed to the Paris Agreement to solve the climate crisis. And that was a precious moment that we have to make sure is not wasted.
    """
    
    spanish_text = """
    Muchas gracias, Chris. Y es en verdad un gran honor tener la oportunidad de venir a este escenario por segunda vez. Estoy extremadamente agradecido. He quedado impresionado por esta conferencia y quiero agradecerles a todos ustedes los muchos comentarios agradables sobre lo que dije la otra noche. Y quiero agradecerles por lo que están haciendo. Aquellos de ustedes que han estado aquí, para aquellos de ustedes que están viendo, las personas que han estado aquí han estado tan activamente involucradas en tratar de hacer de este mundo un lugar mejor. Las brillantes presentaciones que hemos escuchado, las extraordinarias actuaciones que hemos disfrutado, las conversaciones en los pasillos y durante las comidas -- ha sido una experiencia increíble para mí.

    Me voy a centrar en la crisis climática, y tengo una presentación de diapositivas de 10 minutos -- una nueva presentación de diapositivas de 10 minutos -- actualizo la presentación casi todos los días, así que esta es una nueva presentación centrada en la crisis climática. Pero quiero comenzar con un par de historias de las secuelas de la película.

    He estado tratando de contar esta historia durante mucho tiempo. Me lo recordó Tipper, que está justo ahí, ella vio la primera presentación de diapositivas hace 30 años. Y -- oh, hay un escéptico por aquí. Hay algunas personas que han visto la presentación muchas, muchas veces. Pero contar esta historia sobre la crisis climática durante un largo período de tiempo -- hubo un punto de inflexión en 2015 cuando, en diciembre, prácticamente todas las naciones del mundo entero acordaron el Acuerdo de París para resolver la crisis climática. Y ese fue un momento precioso que tenemos que asegurarnos de que no se desperdicie.
    """
    
    # Generate summaries
    # English
    en_basic = basic_summarizer.summarize(english_text, language='en', num_sentences=3)
    print_section("BASIC SUMMARIZER (ENGLISH)", en_basic)
    
    en_tuned = tuned_summarizer.summarize(english_text, language='en', num_sentences=3, title="Climate Crisis TED Talk")
    print_section("DOMAIN-TUNED SUMMARIZER (ENGLISH)", en_tuned)
    
    # Spanish
    es_basic = basic_summarizer.summarize(spanish_text, language='es', num_sentences=3)
    print_section("BASIC SUMMARIZER (SPANISH)", es_basic)
    
    es_tuned = tuned_summarizer.summarize(spanish_text, language='es', num_sentences=3, title="Charla TED sobre la Crisis Climática")
    print_section("DOMAIN-TUNED SUMMARIZER (SPANISH)", es_tuned)

if __name__ == "__main__":
    main()
