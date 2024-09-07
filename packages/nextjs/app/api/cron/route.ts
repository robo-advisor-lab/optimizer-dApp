import { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  console.log("hello world");
  res.status(200).json({ message: "Hello World" });
}
