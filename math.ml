type vector = float list
and matrix = float list list

let dot (a : vector) (b : vector) : float =
  if List.length a <> List.length b then
    Printf.printf "dot: length mismatch: %d vs %d\n%!" (List.length a) (List.length b);
  List.fold_left2 (fun acc x y -> acc +. x *. y) 0.0 a b

let add (a : vector) (b : vector) : vector =
  if List.length a <> List.length b then
    Printf.printf "add: length mismatch: %d vs %d\n%!" (List.length a) (List.length b);
  List.map2 ( +. ) a b

let sigmoid x = 1. /. (1. +. exp (-.x))
let sigmoid_derivative x =
  let s = sigmoid x in
  s *. (1. -. s)

let mat_vec_mul (m : matrix) (v : vector) : vector =
  let v_len = List.length v in
  let row_lengths = List.map List.length m in
  if not (List.for_all ((=) v_len) row_lengths) then begin
    Printf.printf "mat_vec_mul: vector len = %d\n%!" v_len;
    List.iteri (fun i len -> Printf.printf "  row %d len = %d\n%!" i len) row_lengths;
    invalid_arg "mat_vec_mul: all rows must have the same length as the vector"
  end else
    List.map (fun row -> dot row v) m

