//! Another example in span mode, to check and illustrate the use of GLiNER-X (multilingual model).

use gliner::model::output::decoded::SpanOutput;
use orp::params::RuntimeParameters;
use gliner::util::result::Result;
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;


// French example taken from https://huggingface.co/knowledgator/gliner-x-large
const TEXT: &str = "Cristiano Ronaldo dos Santos Aveiro (prononciation portugaise : [kɾiʃ'tjɐnu ʁɔ'naldu] ; né le 5 février 1985) est un footballeur professionnel portugais qui joue comme attaquant et est capitaine à la fois du club d'Al Nassr en Saudi Pro League et de l'équipe nationale du Portugal. Largement considéré comme l’un des plus grands joueurs de tous les temps, Ronaldo a remporté cinq Ballons d’Or, un record de trois prix du Joueur de l’année UEFA et quatre Souliers d’or européens, le plus grand nombre pour un joueur européen. Il a remporté 33 trophées dans sa carrière, dont sept titres de championnat, cinq Ligues des champions de l’UEFA, le Championnat d'Europe et la Ligue des nations de l’UEFA. Ronaldo détient les records du plus grand nombre d'apparitions (183), de buts (140) et de passes décisives (42) en Ligue des champions, de buts dans le Championnat d'Europe (14), de buts internationaux (128) et d'apparitions internationales (205). Il est l’un des rares joueurs à avoir disputé plus de 1 200 matchs professionnels en carrière, le plus grand nombre pour un joueur de champ, et a marqué plus de 850 buts officiels en carrière en club et en sélection, ce qui fait de lui le meilleur buteur de tous les temps.";
const LABELS: [&str; 5] = ["personne", "récompense", "date", "compétitions", "équipes"];
const EXPECTED_ENTITIES: [[&str; 2]; 11] = [
    ["Cristiano Ronaldo dos Santos Aveiro", "personne"],
    ["5 février 1985", "date"],
    ["Al Nassr", "équipes"],
    ["Saudi Pro League", "équipes"],
    ["équipe nationale du Portugal", "équipes"],
    ["Ronaldo", "personne"],
    ["Ballons d’Or", "récompense"],
    ["Championnat d'Europe", "compétitions"],
    ["Ligue des nations de l’UEFA", "compétitions"],
    ["Ligue des champions", "compétitions"],
    ["Championnat d'Europe", "compétitions"],
];


// Use GLiNER-X large in quantized version
const MODEL: &str = "models/gliner-x-large/onnx/model_quantized.onnx"; // https://huggingface.co/knowledgator/gliner-x-large/blob/main/onnx/model_quantized.onnx
const TOKENIZER: &str = "models/gliner-x-large/tokenizer.json"; // https://huggingface.co/knowledgator/gliner-x-large/blob/main/tokenizer.json


// Load the model, perform inference, then check and print the results
fn main() -> Result<()> {    
    println!("Loading model...");
    let model = GLiNER::<SpanMode>::new(Parameters::default(), RuntimeParameters::default(), TOKENIZER, MODEL)?;    

    println!("Inferencing...");
    let input = TextInput::from_str(&[TEXT], &LABELS)?;
    let output = model.inference(input)?;

    println!("Results:\n{output}");    

    Ok(check_results(&output))
}


// Check that the output matches expectations (this code is not efficient, but it doesn't matter here)
fn check_results(output: &SpanOutput) {
    let spans = output.spans.first().expect("expected one sequence");
    assert_eq!(spans.len(), EXPECTED_ENTITIES.len(), "unexpected number of entities");
    for [text, class] in EXPECTED_ENTITIES {
        let span = spans.iter().find(|s| s.text().eq(text)).expect(&format!("missing entity: {}", text));
        assert_eq!(span.class(), class, "unexpected label for entity: {}", text);
    }
}