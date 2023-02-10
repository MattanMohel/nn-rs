use nannou::prelude::*;
use crate::data::mnist::one_hot;
use crate::network::FeedForward;
use crate::matrix::{
    Mat, 
    MatBase
};

const WIDTH:  isize = 28;
const HEIGHT: isize = 28;
const BUFFER: isize = 10;
const PX: isize = 28;
const DRAW_RADIUS: isize = 2;
const DIGIT_MODEL_PATH: &str = "src/models/test";
const IMAGE_MODEL_PATH: &str = "src/models/image";

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
    last_pos: (isize, isize),
    digit_model: FeedForward<5>,
    image_model: FeedForward<4>,
    out: Mat
}

fn to_pixel_xy(x: f32, y: f32) -> (isize, isize) {
    let x = PX*BUFFER / 2 + x as isize;
    let x_p = (WIDTH*HEIGHT / 2 + x) / PX;
    let y_p = (WIDTH*HEIGHT / 2 + y as isize) / PX;

    (x_p, y_p)
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("Sketchbook")
        .size((PX*(WIDTH + BUFFER)) as u32, (PX*HEIGHT) as u32)
        .resizable(false)
        .view(view)
        .mouse_pressed(draw_press)
        .mouse_released(draw_release)
        .key_pressed(key_press)
        .build()
        .unwrap();

    Model {
        buf: [0.0; (WIDTH*HEIGHT) as usize],
        l_mouse_pressed: false,
        r_mouse_pressed: false,
        draw_radius: DRAW_RADIUS,
        last_pos: (-1, -1),
        digit_model: FeedForward::load_model(DIGIT_MODEL_PATH).unwrap(),
        image_model: FeedForward::load_model(IMAGE_MODEL_PATH).unwrap(),
        out: Mat::zeros((10, 1))
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let input = Mat::from_arr(model.buf.map(|n| n as f32));
    model.out = model.digit_model.predict(&input);

    if !(model.l_mouse_pressed || model.r_mouse_pressed) {
        return
    }
    
    let (x, y) = to_pixel_xy(app.mouse.x, app.mouse.y);
    
    if model.last_pos == (x, y) {
        return
    }

    model.last_pos = (x, y);

    for i in -model.draw_radius..model.draw_radius {
        for j in -model.draw_radius..model.draw_radius {
            if let Some(pixel) = model.buf.get_mut((WIDTH*HEIGHT-PX*(y + i) + x + j) as usize) {
                let dist = f32::hypot(i as f32, j as f32);

                if dist >= model.draw_radius as f32 {
                    continue
                }

                if model.l_mouse_pressed {
                    *pixel = (*pixel + 1.0 / dist as f32).clamp(0.0, 1.0);
                }
                else {
                    *pixel = 0.0;
                }
            }
        }
    }
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
        let s = format!("\"{}\": {:.2}%", d, *p * 100.);

        let font_size;

        if i == 0 {
            font_size = 40;
        }
        else {
            font_size = 35;
        }

        draw
            .text(&s)
            .color(BLACK)
            .font_size(font_size)
            .x_y(375., 225. - 1.5 * i as f32 * font_size as f32);
    }

    for (i, p) in model.buf.iter().enumerate() {
        let x = (i % PX as usize) as f32;
        let y = (PX - 1 - i as isize / PX) as f32;

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
        Key::Return => model.buf = [0.0; (WIDTH*HEIGHT) as usize],
        _ if key as usize <= 9 => {
            let digit = (key as usize + 1) % 10;
            model.buf = model.image_model
                .predict(&one_hot(digit))
                .data()
                .clone()
                .try_into()
                .unwrap()
        }
        Key::Up => {
            if model.draw_radius <= 5 {
                model.draw_radius += 1
            }
        }
        Key::Down => {
            if model.draw_radius > 1 {
                model.draw_radius -= 1
            }
        }
        _ => ()
    }
}
