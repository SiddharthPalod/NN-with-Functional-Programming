open Math

let add_debug_map2 f a b label =
  if List.length a <> List.length b then begin
    Printf.printf "%s: length mismatch: %d vs %d\n%!" label (List.length a) (List.length b);
    failwith ("List.map2 failure at " ^ label)
  end else
    List.map2 f a b


let add_debug_fold_left2 f acc a b label =
  if List.length a <> List.length b then
    Printf.printf "%s: length mismatch: %d vs %d\n%!" label (List.length a) (List.length b);
  List.fold_left2 f acc a b

type nn_params = {
  w1 : matrix;  (* 76x4 weights *)
  b1 : vector;  (* 76 biases *)
  w2 : matrix;  (* 72x76 weights *)
  b2 : vector;  (* 72 biases *)
  w3 : matrix;  (* 68x72 weights *)
  b3 : vector;  (* 68 biases *)
  w4 : matrix;  (* 1x68 weights *)
  b4 : vector;  (* 1 bias *)
}

let random_matrix rows cols =
  let rand () = (Random.float 2. -. 1.) *. 0.1 in
  List.init rows (fun _ -> List.init cols (fun _ -> rand ()))

let random_vector n = List.init n (fun _ -> (Random.float 2. -. 1.) *. 0.1)

let init_params () = {
  w1 = random_matrix 76 4;
  b1 = random_vector 76;
  w2 = random_matrix 72 76;
  b2 = random_vector 72;
  w3 = random_matrix 68 72;
  b3 = random_vector 68;
  w4 = random_matrix 1 68;
  b4 = random_vector 1;
}

let relu x = if x > 0. then x else 0.
let relu_layer xs = List.map relu xs
let relu_derivative x = if x > 0. then 1. else 0.

let dropout_layer xs rate =
  let keep_prob = 1.0 -. rate in
  List.map (fun x ->
    if Random.float 1.0 < keep_prob then x /. keep_prob else 0.0
  ) xs

let forward_pass ?(train=false) ?(dropout_rate=0.2) (params : nn_params) (input : vector) : (vector * vector * vector * vector * vector) =
  let z1 = add (mat_vec_mul params.w1 input) params.b1 in
  let a1 = relu_layer z1 in
  let a1 = if train then dropout_layer a1 dropout_rate else a1 in
  let z2 = add (mat_vec_mul params.w2 a1) params.b2 in
  let a2 = relu_layer z2 in
  let a2 = if train then dropout_layer a2 dropout_rate else a2 in
  let z3 = add (mat_vec_mul params.w3 a2) params.b3 in
  let a3 = relu_layer z3 in
  let a3 = if train then dropout_layer a3 dropout_rate else a3 in
  let z4 = add (mat_vec_mul params.w4 a3) params.b4 in
  let a4 = List.map sigmoid z4 in
  (a1, a2, a3, a4, z4)

let binary_cross_entropy y_true y_pred =
  let eps = 1e-15 in
  let y_pred = max eps (min (1. -. eps) y_pred) in
  -. (y_true *. log y_pred +. (1. -. y_true) *. log (1. -. y_pred))

let backprop (params : nn_params) (x : vector) (y : float) : nn_params =
  let z1 = add (mat_vec_mul params.w1 x) params.b1 in
  let a1 = relu_layer z1 in
  let z2 = add (mat_vec_mul params.w2 a1) params.b2 in
  let a2 = relu_layer z2 in
  let z3 = add (mat_vec_mul params.w3 a2) params.b3 in
  let a3 = relu_layer z3 in
  let z4 = add (mat_vec_mul params.w4 a3) params.b4 in
  let a4 = List.map sigmoid z4 in
  let y_hat = List.hd a4 in

  let dz4 = [y_hat -. y] in
  let dw4 = List.map (fun d -> List.map (fun a -> d *. a) a3) dz4 in
  let db4 = dz4 in

  let w4_t = List.init (List.length (List.hd params.w4)) (fun i -> List.map (fun row -> List.nth row i) params.w4) in
  let da3 = mat_vec_mul w4_t dz4 in
  let dz3 = List.map2 (fun d z -> d *. relu_derivative z) da3 z3 in
  let dw3 = List.map (fun dz -> List.map (fun a -> dz *. a) a2) dz3 in
  let db3 = dz3 in

  let w3_t = List.init (List.length (List.hd params.w3)) (fun i -> List.map (fun row -> List.nth row i) params.w3) in
  let da2 = mat_vec_mul w3_t dz3 in
  let dz2 = List.map2 (fun d z -> d *. relu_derivative z) da2 z2 in
  let dw2 = List.map (fun dz -> List.map (fun a -> dz *. a) a1) dz2 in
  let db2 = dz2 in

  let w2_t = List.init (List.length (List.hd params.w2)) (fun i -> List.map (fun row -> List.nth row i) params.w2) in
  let da1 = mat_vec_mul w2_t dz2 in
  let dz1 = List.map2 (fun d z -> d *. relu_derivative z) da1 z1 in
  let dw1 = List.map (fun dz -> List.map (fun xi -> dz *. xi) x) dz1 in
  let db1 = dz1 in

  {
    w1 = dw1;
    b1 = db1;
    w2 = dw2;
    b2 = db2;
    w3 = dw3;
    b3 = db3;
    w4 = dw4;
    b4 = db4;
  }

let update_params params grads lr =
  let update_matrix m g =
    add_debug_map2 (fun row g_row -> add_debug_map2 (fun w dw -> w -. lr *. dw) row g_row "update_matrix-inner") m g "update_matrix-outer"
  in
  let update_vector v g = 
  add_debug_map2 (fun w dw -> w -. lr *. dw) v g "update_vector" in
  {
    w1 = update_matrix params.w1 grads.w1;
    b1 = update_vector params.b1 grads.b1;
    w2 = update_matrix params.w2 grads.w2;
    b2 = update_vector params.b2 grads.b2;
    w3 = update_matrix params.w3 grads.w3;
    b3 = update_vector params.b3 grads.b3;
    w4 = update_matrix params.w4 grads.w4;
    b4 = update_vector params.b4 grads.b4;
  }

let compute_loss params dataset =
  let total_loss =
    List.fold_left (fun acc (x, y) ->
      let (_a1, _a2, _a3, _a4, z4) = forward_pass params x in
      let y_hat = List.map sigmoid z4 |> List.hd in
      acc +. binary_cross_entropy y y_hat
    ) 0.0 dataset
  in
  total_loss /. float_of_int (List.length dataset)

let split_at n lst =
  let rec aux i acc rest =
    if i = 0 then (List.rev acc, rest)
    else match rest with
      | [] -> (List.rev acc, [])
      | x :: xs -> aux (i - 1) (x :: acc) xs
  in
  aux n [] lst


let shuffle lst =
  let rnd_int n = Random.int n in
  let rec aux acc = function
    | [] -> List.rev acc
    | h :: t ->
        let i = rnd_int (List.length acc + 1) in
        let (left, right) = split_at i acc in
        aux (List.append left (h :: right)) t
  in
  aux [] lst

let sublist lst start len =
  let rec aux i acc = function
    | [] -> List.rev acc
    | h :: t ->
        if i >= start && i < start + len then aux (i + 1) (h :: acc) t
        else aux (i + 1) acc t
  in
  aux 0 [] lst

let take_n lst n =
  let rec aux acc i l =
    match l with
    | [] -> (List.rev acc, [])
    | x :: xs ->
      if i = 0 then (List.rev acc, l)
      else aux (x :: acc) (i - 1) xs
  in
  aux [] n lst

let split_into_batches lst batch_size =
  let rec aux acc lst =
    match lst with
    | [] -> List.rev acc
    | _ ->
      let batch, rest = take_n lst batch_size in
      aux (batch :: acc) rest
  in
  aux [] lst



let train ?(learning_rate=0.01) ?(dropout_rate=0.3) ?(patience=10) ?(min_delta=1e-4) ?(batch_size=48) params trainset testset epochs =
  let rec loop n p acc best_p best_loss epochs_no_improve =
    if n = 0 || epochs_no_improve >= patience then (best_p, List.rev acc)
    else
      let trainset_shuffled = shuffle trainset in
      let batches = split_into_batches trainset_shuffled batch_size in
      let p' =
        List.fold_left (fun p batch ->
          List.fold_left (fun p (x, y) ->
            let grads = backprop p x y in
            update_params p grads learning_rate
          ) p batch
        ) p batches
      in
      let train_loss =
        List.fold_left (fun acc (x, y) ->
          let (_a1, _a2, _a3, a4, _z4) = forward_pass ~train:true ~dropout_rate p' x in
          let y_pred = List.hd a4 in
          acc +. binary_cross_entropy y y_pred
        ) 0.0 trainset /. float_of_int (List.length trainset)
      in
      let test_loss =
        List.fold_left (fun acc (x, y) ->
          let (_a1, _a2, _a3, a4, _z4) = forward_pass p' x in
          let y_pred = List.hd a4 in
          acc +. binary_cross_entropy y y_pred
        ) 0.0 testset /. float_of_int (List.length testset)
      in
      let train_acc =
        let correct = List.fold_left (fun acc (x, y) ->
          let (_a1, _a2, _a3, a4, _z4) = forward_pass ~train:true ~dropout_rate p' x in
          let y_pred = List.hd a4 in
          let y_bin = if y_pred >= 0.5 then 1.0 else 0.0 in
          if y_bin = y then acc + 1 else acc
        ) 0 trainset in
        float_of_int correct /. float_of_int (List.length trainset)
      in
      let test_acc =
        let correct = List.fold_left (fun acc (x, y) ->
          let (_a1, _a2, _a3, a4, _z4) = forward_pass p' x in
          let y_pred = List.hd a4 in
          let y_bin = if y_pred >= 0.5 then 1.0 else 0.0 in
          if y_bin = y then acc + 1 else acc
        ) 0 testset in
        float_of_int correct /. float_of_int (List.length testset)
      in
      let entry = (epochs - n + 1, train_loss, test_loss, train_acc, test_acc) in
      let (best_p, best_loss, epochs_no_improve) =
        if test_loss < best_loss -. min_delta then (p', test_loss, 0)
        else (best_p, best_loss, epochs_no_improve + 1)
      in
      loop (n - 1) p' (entry :: acc) best_p best_loss epochs_no_improve
  in
  let initial_loss = max_float in
  loop epochs params [] params initial_loss 0
