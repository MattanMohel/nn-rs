use nannou::prelude::*;
use crate::data::dataset::Dataset;
use crate::data::mnist::Reader;
use crate::network::Net;
use crate::matrix::{
    Mat, 
    MatBase
};

const WIDTH:  u32 = 28;
const HEIGHT: u32 = 28;
const BUFFER: u32 = 10;
const PX: u32 = 28;
const DRAW_RADIUS: isize = 2;
const MODEL_PATH: &str = "src/models/test";

pub fn run_sketch() {
    nannou::app(model)
        .update(update)
        .run()
}

struct Model {
    buf: [f32; 784],
    l_mouse_pressed: bool,
    r_mouse_pressed: bool,
    draw_radius: isize,
    net: Net<4>,
    out: Mat
}

fn to_pixel_xy(x: f32, y: f32) -> (f32, f32) {
    let x = x + (PX*BUFFER / 2) as f32;

    let x_p = (784.0 / 2.0 + x) / 28.0;
    let x_p = x_p.floor();

    let y_p = (784.0 / 2.0 + y) / 28.0;
    let y_p = y_p.floor();

    (x_p, y_p)
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("Sketchbook")
        .size(PX*(WIDTH + BUFFER), PX*HEIGHT)
        .resizable(false)
        .view(view)
        .mouse_pressed(draw_press)
        .mouse_released(draw_release)
        .key_pressed(key_press)
        .build()
        .unwrap();

    Model {
        buf: [0.0; 784],
        l_mouse_pressed: false,
        r_mouse_pressed: false,
        draw_radius: DRAW_RADIUS,
        net: Net::from_file(MODEL_PATH),
        out: Mat::zeros((10, 1))
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    if !(model.l_mouse_pressed || model.r_mouse_pressed) {
        return;
    }
    
    let (x, y) = to_pixel_xy(app.mouse.x, app.mouse.y);
    if x >= (WIDTH + BUFFER) as f32 || x < 0.0 || y >= HEIGHT as f32 || y < 0.0 {
        return;
    }

    for i in -model.draw_radius..model.draw_radius {
        for j in -model.draw_radius..model.draw_radius {
            let i = i as f32;
            let j = j as f32;

            let dist = (i.powi(2) + j.powi(2)).sqrt();

            if dist >= model.draw_radius as f32 {
                continue;
            }

            let cell = model.buf.get_mut((784.0-28.0*(y + i) + (x + j)) as usize);

            if let Some(pixel) = cell {
                if model.l_mouse_pressed {
                    *pixel += (1.0 / dist.powi(2) as f32).clamp(0.0, 1.0);
                }
                else {
                    *pixel = 0.0;
                }
            }
        }
    }

    let vec = model.buf.map(|n| n as f32 / 5.0);
    let mat = Mat::from_vec((784, 1), vec.to_vec());
    model.out = model.net.predict(&mat);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();     
    draw.background().color(WHITESMOKE);

    let mut out: Vec<_> = model.out
        .data()
        .iter()
        .enumerate()
        .collect();

    out.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (i, (d, p)) in out.iter().enumerate() {
        let s = format!("\"{}\": {}%", d, p);
        draw
            .text(&s)
            .color(BLACK)
            .font_size(25)
            .x_y(400., 150. - i as f32 * 50.);
    }

    for (i, p) in model.buf.iter().enumerate() {
        let x = (i % 28) as f32;
        let y = 28.0 - (i / 28) as f32;

        if *p > 0.0 {
            let r = Rect::from_w_h(28.0, 28.0)
                .bottom_left_of(app.window_rect())
                .shift_x(PX as f32 * x)
                .shift_y(PX as f32 * y);
        
            draw.rect()
                .wh(r.wh())
                .xy(r.xy())
                .color(rgba(0.0, 0.0, 0.0, *p));
        }
    }

    draw.to_frame(app, &frame).unwrap();
}

fn draw_press(_: &App, model: &mut Model, mouse: MouseButton) {
    match mouse {
        MouseButton::Left =>   model.l_mouse_pressed = true,
        MouseButton::Right =>  model.r_mouse_pressed = true,
        _ => ()
    }
}

fn draw_release(_: &App, model: &mut Model, mouse: MouseButton) {
    match mouse {
        MouseButton::Left =>   model.l_mouse_pressed = false,
        MouseButton::Right =>  model.r_mouse_pressed = false,
        _ => ()
    }
}

fn key_press(_: &App, model: &mut Model, key: Key) {
    match key {
        Key::R => model.buf = [0.0; 784],
        Key::Up => model.draw_radius += 1,
        Key::Down => {
            if model.draw_radius > 1 {
                model.draw_radius -= 1
            }
        }
        Key::P => {
            let buf = model.buf.map(|n| n as f32);
            let buf = Mat::from_vec((784, 1), buf.to_vec());

            for (i, byte) in buf.data().iter().enumerate() {
                if *byte > 0.8 {
                    print!("■ ")
                } else if *byte > 0.4 {
                    print!("▧ ")
                } else if *byte > 0.0 {
                    print!("□ ")
                } else {
                    print!("- ")
                };
    
                if i % 28 == 0 {
                    println!()
                } 
            }

            let digit = model.net.predict(&buf).max_index().0;

            println!("predicted digit: {}", digit);
        }
        _ => ()
    }
}
