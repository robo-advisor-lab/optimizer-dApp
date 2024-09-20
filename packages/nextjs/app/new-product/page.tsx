"use client";

import React from "react";
import { useState } from "react";
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select, { SelectChangeEvent } from "@mui/material/Select";
import TextField from "@mui/material/TextField";

const Page = () => {
  const [durationForm, setDurationForm] = useState("");
  const [frequenceForm, setFrequenceForm] = useState("");
  const [numAdress, setNumAdres] = useState(1);
  const elements = [];

  const handleClickAdressInputPlus = () => {
    setNumAdres(numAdress + 1);
  };
  const handleClickAdressInputLesss = () => {
    if (numAdress > 1) setNumAdres(numAdress - 1);
  };

  const handleChangeDuration = (event: SelectChangeEvent) => {
    setDurationForm(event.target.value as string);
  };

  const handleChangeFrequence = (event: SelectChangeEvent) => {
    setFrequenceForm(event.target.value as string);
  };

  for (let index = 0; index < numAdress; index++) {
    elements.push(
      <Box component="form" sx={{ "& > :not(style)": { m: 1, width: "25ch" } }} noValidate autoComplete="off">
        <TextField
          id="outlined-basic"
          label={"Cripto Address " + (index + 1)}
          variant="outlined"
          name={"adress" + index}
        />
      </Box>,
    );
  }
  return (
    <div>
      <div className="container mx-auto px-4 py-16">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="md:w-1/2 mb-8 md:mb-0">
            <h2 className="text-4xl font-bold mb-4">DAO Governance Parameters</h2>

            <div className="dao-governance-section">
              <p>The DAO can vote on the following parameters for each investment vehicle issuance:</p>

              <ul className="parameter-list">
                <li className="mb-4">
                  <strong className="mb-4">Tduration:</strong> Duration of the vehicle (e.g., 6 months, 1 year, 10
                  years)
                </li>
                <li className="mb-4">
                  <strong>Trebalancing:</strong> Rebalancing frequency (e.g., hourly, daily, weekly, yearly)
                </li>
                <li className="mb-4">
                  <strong>ntokens:</strong> Number of tokens to include in the vehicle (e.g., 2, 3, ...)
                </li>
                <li className="mb-4">
                  <strong>Ai:</strong> Set of token addresses to include in the vehicle, where i ∈ 1, 2, ..., ntokens
                </li>
              </ul>

              <p>
                These parameters allow the DAO to adjust the characteristics of each investment vehicle to adapt to
                different strategies and market conditions.
              </p>

              <h3 className="subsection-title">Additional Considerations</h3>

              <ul className="considerations-list">
                <li>
                  <strong>Governance:</strong> The DAO can adjust the token creation/destruction criteria through
                  voting.
                </li>
                <li>
                  <strong>Fees:</strong> The SC may charge fees for fund management and rebalancing, which will be
                  deducted from the NAV.
                </li>
              </ul>
            </div>
          </div>
          <div className="md:w-1/2">
            <div className="bg-blue-100 px-6 py-4 mt-3 rounded-lg">
              <h2 className="text-4xl font-bold mb-4">DAO Governance Parameters</h2>
              <p>Select cryptocurrencies</p>
              <FormGroup>
                <FormControlLabel control={<Checkbox defaultChecked />} label="SOL" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="ETH" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="USDC" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="TON" />
              </FormGroup>
              <div className="flex gap-3">
                <p>Address</p>
                <div
                  className="bg-blue-300 text-blue-800 px-3 rounded-full flex items-center justify-center cursor-pointer select-none"
                  onClick={handleClickAdressInputPlus}
                >
                  +
                </div>
                <div
                  className="bg-red-300 text-red-800 px-3 rounded-full flex items-center justify-center cursor-pointer select-none"
                  onClick={handleClickAdressInputLesss}
                >
                  -
                </div>
              </div>
              <div>{elements}</div>
              <p>Duration</p>
              <Box sx={{ minWidth: 120 }}>
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Duration</InputLabel>
                  <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={durationForm}
                    label="Duracion"
                    onChange={handleChangeDuration}
                  >
                    <MenuItem value={10}>Ten</MenuItem>
                    <MenuItem value={20}>Twenty</MenuItem>
                    <MenuItem value={30}>Thirty</MenuItem>
                  </Select>
                </FormControl>
              </Box>
              <p>Rebalancing frequency</p>
              <Box sx={{ minWidth: 120 }}>
                <FormControl fullWidth>
                  <InputLabel id="demo-simple-select-label">Rebalancing frequency</InputLabel>
                  <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={frequenceForm}
                    label="Frecuencia"
                    onChange={handleChangeFrequence}
                  >
                    <MenuItem value={10}>Ten</MenuItem>
                    <MenuItem value={20}>Twenty</MenuItem>
                    <MenuItem value={30}>Thirty</MenuItem>
                  </Select>
                </FormControl>
              </Box>
            </div>
          </div>
        </div>

        {/* Additional sections can be added here for features, benefits, testimonials, etc. */}
      </div>
    </div>
  );
};

export default Page;
