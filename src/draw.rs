use nannou::prelude::*;

use crate::network::{ Net, Mat };

const COL: usize = 784;
const ROW: usize = 784;
const COL_P: usize = 28;
const ROW_P: usize = 28;
const DIM_P: usize = COL_P*ROW_P;

const MODEL_PATH: &str = "src/models/model";

pub fn run_sketch() {
    nannou::app(model)
        .update(update)
        .run()
}

struct Model {
    buf: [f32; DIM_P],
    l_mouse_pressed: bool,
    r_mouse_pressed: bool,
    net: Net<4>
}

fn to_pixel_xy(x: f32, y: f32) -> (f32, f32) {
    let x_p = (COL as f32 / 2.0 + x) / (COL / COL_P) as f32;
    let x_p = x_p.floor();

    let y_p = (ROW as f32 / 2.0 - y) / (ROW / ROW_P) as f32;
    let y_p = y_p.floor();

    (x_p, y_p)
}

fn model(app: &App) -> Model {
    app.new_window()
        .title("test")
        .resizable(false)
        .size(COL as u32, ROW as u32)
        .view(view)
        .mouse_pressed(draw_press)
        .mouse_released(draw_release)
        .key_pressed(reset_buffer)
        .build()
        .unwrap();

    Model {
        buf: [0.0; DIM_P],
        l_mouse_pressed: false,
        r_mouse_pressed: false,
        net: Net::from_file(MODEL_PATH)
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    if !(model.l_mouse_pressed || model.r_mouse_pressed) {
        return;
    }
    
    let (x, y) = to_pixel_xy(app.mouse.x, app.mouse.y);

    if x >= COL_P as f32 || x < 0.0 || y >= ROW_P as f32 || y < 0.0 {
        return;
    }

    let state = if model.l_mouse_pressed {
        1.0
    }
    else {
        0.0
    };

    for i in -1..=1 {
        for j in -1..=1 {
            if i.abs() == 1 && j.abs() == 1 {
                continue;
            }

            match model.buf.get_mut(ROW_P * (y as i32 + i) as usize + (x + j as f32) as usize) {
                Some(p) => *p = state,
                _ => ()
            }
        }
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    // get pixel relative draw frame
    let draw = app
        .draw()
        .scale_x(COL as f32 / COL_P as f32)
        .scale_y(ROW as f32 / ROW_P as f32)
        .x_y(0.5 - (COL_P as f32) / 2.0, (ROW_P as f32) / 2.0 - 0.5);

    draw.background().color(SNOW);
 
    for (i, p) in model.buf.iter().enumerate() {
        let x = (i % COL_P) as f32;
        let y = (i / ROW_P) as f32;

        if *p > 0.0 {
            draw.rect()
                .color(rgba(0.0, 0.0, 0.0, *p))
                .w_h(1.0, 1.0)
                .x_y(x, -y);
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

fn reset_buffer(_: &App, model: &mut Model, key: Key) {
    match key {
        Key::R => model.buf = [0.0; DIM_P],
        Key::P => {
            let buf = model.buf.map(|n| n as f32);
            let buf = Mat::from_vec(DIM_P, 1, buf.to_vec());

            let digit = model.net.predict(&buf);

            let mut m = -100.;
            let mut i = 0;
            for j in 0..10 {
                if m < digit[j] {
                    m = digit[j];
                    i = j
                }
            }

            println!("the digit is {} with {} certainty", i, m);
        }
        _ => ()
    }
}
