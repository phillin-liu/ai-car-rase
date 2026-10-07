// Anti-cheat batch validator for AI Car Race.
//
// Reads a JSON file containing a list of raw match result dicts (the same
// shape produced by `MatchResult.to_dict()`) and validates physical
// plausibility of the recorded telemetry:
//
//   * finite speed / distance values
//   * speed below the theoretical maximum (plus a small tolerance)
//   * no negative collision counters
//   * finished cars must have a sane finish time
//
// Exit code 0 always; the verdict is printed as JSON to stdout.
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;

#[derive(Deserialize)]
struct Match {
    #[serde(default)]
    cars: Vec<Car>,
}

#[derive(Deserialize)]
struct Car {
    name: Option<String>,
    finished: Option<bool>,
    finish_time: Option<f64>,
    avg_speed: Option<f64>,
    max_speed: Option<f64>,
    distance: Option<f64>,
    wall_hits: Option<f64>,
    obstacle_hits: Option<f64>,
    hazard_hits: Option<f64>,
    progress: Option<f64>,
}

fn f(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ac-validate <matches.json>");
        std::process::exit(2);
    }
    let text = match fs::read_to_string(&args[1]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("read error: {e}");
            std::process::exit(2);
        }
    };
    let payload: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("json error: {e}");
            std::process::exit(2);
        }
    };
    let matches: Vec<Match> = serde_json::from_value(
        payload.get("matches").cloned().unwrap_or(serde_json::Value::Null),
    )
    .unwrap_or_default();

    let mut violations: Vec<serde_json::Value> = Vec::new();

    for m in &matches {
        for (idx, car) in m.cars.iter().enumerate() {
            let side = car
                .name
                .clone()
                .unwrap_or_else(|| format!("P{}", idx + 1));

            let avg = f(car.avg_speed);
            let maxs = f(car.max_speed);
            let dist = f(car.distance);
            let wall = f(car.wall_hits);
            let obs = f(car.obstacle_hits);
            let haz = f(car.hazard_hits);
            let prog = f(car.progress);
            let ftime = f(car.finish_time);

            if !avg.is_finite() || avg < 0.0 {
                violations.push(serde_json::json!({
                    "side": side, "code": "non_finite", "detail": "avg_speed invalid"
                }));
            }
            if !maxs.is_finite() || maxs > 300.0 {
                violations.push(serde_json::json!({
                    "side": side, "code": "speed_cap", "detail": format!("max_speed={maxs}")
                }));
            }
            if !dist.is_finite() || dist < 0.0 {
                violations.push(serde_json::json!({
                    "side": side, "code": "non_finite", "detail": "distance invalid"
                }));
            }
            if !wall.is_finite() || wall < 0.0 || !obs.is_finite() || obs < 0.0
                || !haz.is_finite() || haz < 0.0
            {
                violations.push(serde_json::json!({
                    "side": side, "code": "negative_hits", "detail": "negative collision counters"
                }));
            }
            if !prog.is_finite() || prog < 0.0 {
                violations.push(serde_json::json!({
                    "side": side, "code": "bad_progress", "detail": format!("progress={prog}")
                }));
            }
            if car.finished == Some(true) && (!ftime.is_finite() || ftime < 0.0) {
                violations.push(serde_json::json!({
                    "side": side, "code": "bad_finish", "detail": "finished but finish_time invalid"
                }));
            }
        }
    }

    let mut by_code: HashMap<String, u32> = HashMap::new();
    for v in &violations {
        let code = v["code"].as_str().unwrap_or("unknown").to_string();
        *by_code.entry(code).or_insert(0) += 1;
    }

    let details: Vec<serde_json::Value> = if violations.len() > 20 {
        violations[..20].to_vec()
    } else {
        violations.clone()
    };

    let verdict = serde_json::json!({
        "engine": "rust",
        "violations": violations.len(),
        "by_code": by_code,
        "details": details,
    });
    println!("{}", serde_json::to_string_pretty(&verdict).unwrap());
}
