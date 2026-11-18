use super::*;
use ort::value::Tensor;

const MODEL: &[u8] = include_bytes!("../../../tests/fixtures/concurrent_logits.onnx");

fn input(value: i64) -> Result<SessionInputs<'static, 'static>> {
    Ok(ort::inputs![
        "input_ids" => Tensor::from_array(([1usize, 1], vec![value]))?,
        "attention_mask" => Tensor::from_array(([1usize, 1], vec![1i64]))?,
    ].into())
}

#[test]
fn owned_outputs_survive_subsequent_and_concurrent_runs() -> Result<()> {
    let model = Model::new_from_bytes(MODEL, RuntimeParameters::default().with_threads(1))?;
    let first = model.run(input(10)?)?;
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|worker| {
                let model = &model;
                let barrier = &barrier;
                scope.spawn(move || -> Result<()> {
                    barrier.wait();
                    for request in 0..32 {
                        let value = i64::from(worker * 32 + request);
                        let outputs = model.run(input(value)?)?;
                        assert_eq!(outputs["logits"].try_extract_tensor::<f32>()?.1, &[(value + 1) as f32]);
                    }
                    Ok(())
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("native inference worker must not panic")?;
        }
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
    })?;
    assert_eq!(first["logits"].try_extract_tensor::<f32>()?.1, &[11.0]);
    Ok(())
}
